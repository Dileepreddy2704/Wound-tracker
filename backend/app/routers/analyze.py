import json
import os

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas import schemas
from app.models.segmentation import WoundSegmenter
from app.models.measurement import mask_area_px, px_to_cm2, area_change_pct, classify_trend

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Loaded once at import time. Set MEDSAM_CHECKPOINT_PATH in .env once you've
# downloaded a checkpoint via ml/download_medsam_checkpoint.py — until then
# this falls back to the stub segmenter so the API stays testable.
_checkpoint_path = os.getenv("MEDSAM_CHECKPOINT_PATH")
if _checkpoint_path and not os.path.exists(_checkpoint_path):
    print(
        f"WARNING: MEDSAM_CHECKPOINT_PATH set but file not found at "
        f"{_checkpoint_path}; using stub segmenter."
    )
    _checkpoint_path = None

segmenter = WoundSegmenter(checkpoint_path=_checkpoint_path)


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post("/{visit_id}", response_model=schemas.WoundAnalysisOut)
def analyze_visit(visit_id: str, db: Session = Depends(get_db)):
    visit = db.query(models.Visit).filter_by(id=visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Visit not found")

    image    = Image.open(visit.image_path).convert("RGB")
    image_np = np.array(image)

    # 1. Segmentation
    mask    = segmenter.segment(image)
    area_px = mask_area_px(mask)

    # Save the mask so Streamlit and clinicians can visually verify it.
    mask_path = visit.image_path.rsplit(".", 1)[0] + "_mask.png"
    Image.fromarray((mask * 255).astype("uint8")).save(mask_path)

    # 2. Wound area in cm²
    #    Priority: (a) user-supplied reference object, (b) EXIF DPI fallback.
    area_cm2 = None
    if visit.reference_object_diameter_cm:
        # TODO: replace placeholder_ref_px with detected reference-object size
        # once reference-object detection is wired in.
        placeholder_ref_px = 50.0
        area_cm2 = px_to_cm2(
            area_px,
            visit.reference_object_diameter_cm,
            placeholder_ref_px,
        )
    if area_cm2 is None:
        area_cm2 = _estimate_cm2_from_dpi(image, area_px)

    # 3. Tissue classification (HSV color analysis inside the wound mask)
    tissue_type, tissue_confidence, tissue_composition = _classify_tissue(image_np, mask)

    # 4. Infection risk (rule-based, derived from full tissue composition + wound size)
    infection_risk_flag, infection_indicators = _assess_infection_risk(
        tissue_type, tissue_confidence, area_cm2, tissue_composition
    )

    # 5. Compare with previous visit for the same patient
    prev_visit = (
        db.query(models.Visit)
        .filter(
            models.Visit.patient_id == visit.patient_id,
            models.Visit.visit_date < visit.visit_date,
        )
        .order_by(models.Visit.visit_date.desc())
        .first()
    )
    change_pct, trend = None, None
    if prev_visit and prev_visit.analysis and prev_visit.analysis.area_cm2 and area_cm2:
        change_pct = area_change_pct(area_cm2, prev_visit.analysis.area_cm2)
        trend      = classify_trend(change_pct)

    # 6. Wound type classification (morphology-based shape analysis)
    wound_type, wound_type_confidence = _classify_wound_type(mask)

    # 7. Clinical report
    report_text = _build_report(
        area_cm2,
        wound_type,
        tissue_type,
        tissue_confidence,
        tissue_composition,
        infection_risk_flag,
        infection_indicators,
        trend,
        change_pct,
    )

    analysis = models.WoundAnalysis(
        visit_id=visit.id,
        mask_path=mask_path,
        area_px=area_px,
        area_cm2=area_cm2,
        tissue_type=tissue_type,
        tissue_confidence=tissue_confidence,
        wound_type=wound_type,
        wound_type_confidence=wound_type_confidence,
        infection_risk_flag=infection_risk_flag,
        infection_indicators=infection_indicators,
        area_change_pct=change_pct,
        healing_trend=trend,
        report_text=report_text,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


# ---------------------------------------------------------------------------
# Tissue classification
# ---------------------------------------------------------------------------

def _classify_tissue(
    image_np: np.ndarray, mask: np.ndarray
) -> tuple[str, float, dict]:
    """
    Classify wound tissue type using HSV color analysis of pixels inside the
    segmentation mask.

    Categories:
      - granulation  : bright red/pink vascular tissue — sign of active healing
      - slough       : yellow/white fibrinous debris   — non-viable, impedes healing
      - necrosis     : dark brown/black dead tissue    — highest risk, requires debridement
      - mixed        : no single category clearly dominates (< 40 % share)

    Returns (dominant_type, confidence, composition_dict).
    composition_dict contains percentage breakdown across all tissue types,
    enabling nuanced risk assessment even when one type dominates.

    This is a rule-based heuristic, not a trained classifier.
    Accuracy depends on lighting conditions and skin tone.
    """
    wound_px = image_np[mask.astype(bool)]
    if len(wound_px) == 0:
        return "unclassified", 0.0, {}

    # Convert wound pixels to HSV for reliable colour thresholding.
    hsv = cv2.cvtColor(
        wound_px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3).astype(np.float32)

    h = hsv[:, 0]   # Hue  0–179 (OpenCV)
    s = hsv[:, 1]   # Sat  0–255
    v = hsv[:, 2]   # Val  0–255
    n = len(h)

    # Granulation tissue: vivid red, moderate-to-high brightness.
    # Hue wraps at 0/179 — both very low and very high hue values = red.
    gran = ((h <= 12) | (h >= 168)) & (s >= 75) & (v >= 55)

    # Necrotic tissue: very dark pixels, or dark brownish tones.
    necro = (v < 65) | ((h >= 8) & (h <= 30) & (s >= 40) & (v < 120))

    # Slough / fibrin: warm yellow hue with decent brightness,
    # OR pale/desaturated bright (white fibrin coating / macerated edges).
    slough = ((h > 12) & (h < 50) & (v >= 100)) | ((s < 55) & (v >= 140))

    pcts = {
        "granulation": float(gran.sum()) / n,
        "necrosis":    float(necro.sum()) / n,
        "slough":      float(slough.sum()) / n,
    }

    dominant = max(pcts, key=pcts.get)
    conf     = pcts[dominant]

    # If no single category clearly dominates, report as "mixed".
    if conf < 0.40:
        return "mixed", round(max(pcts.values()), 2), pcts
    return dominant, round(conf, 2), pcts


# ---------------------------------------------------------------------------
# Wound type classification
# ---------------------------------------------------------------------------

def _classify_wound_type(mask: np.ndarray) -> tuple[str, float]:
    """
    Classify the wound type from the shape and morphology of the segmentation mask.

    Types (matching standard clinical categories):
      - puncture   : small, circular / oval deep hole  (nail, needle)
      - incision   : long, narrow, clean-edged linear cut  (scalpel, knife)
      - laceration : irregular, jagged tear  (blunt trauma)
      - abrasion   : wide, shallow scrape  (friction)
      - avulsion   : flap of tissue torn away  (high-energy trauma)
      - burn        : large, irregular, often with diffuse boundary

    Shape descriptors used:
      - Circularity      = 4π·Area / Perimeter²  (1.0 = perfect circle)
      - Aspect ratio     = major axis / minor axis  (fitted ellipse)
      - Solidity         = Area / Convex Hull Area  (1.0 = smooth convex)
      - Relative area    = wound pixels / total image pixels
      - Extent           = Area / Bounding Box Area

    Note: this is a morphology heuristic — accuracy is limited for
    irregular or partially-occluded wounds. A trained classifier
    would significantly outperform these rules.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "unknown", 0.0

    # Use the largest contour (the primary wound boundary)
    contour = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(contour)

    if area < 100:
        return "unknown", 0.0

    perimeter = cv2.arcLength(contour, closed=True)

    # ── Shape descriptors ────────────────────────────────────────────────
    # Circularity: 1.0 = perfect circle; lower → more elongated/irregular
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Solidity: 1.0 = smooth convex boundary; lower → jagged / irregular
    hull      = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0.0

    # Aspect ratio from fitted ellipse (more stable than bounding box)
    if len(contour) >= 5:
        _, (minor_ax, major_ax), _ = cv2.fitEllipse(contour)
        aspect_ratio = major_ax / minor_ax if minor_ax > 0 else 1.0
    else:
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

    # Extent: how much of the bounding box the wound fills
    x, y, bw, bh = cv2.boundingRect(contour)
    extent = area / (bw * bh) if (bw * bh) > 0 else 0.0

    # Relative size of the wound in the full image
    relative_area = area / mask.size

    # ── Classification rules (order = priority) ──────────────────────────

    # Puncture: small + circular
    if relative_area < 0.015 and circularity > 0.60:
        conf = round(min(circularity, 1.0), 2)
        return "puncture", conf

    # Incision: elongated + smooth edges
    if aspect_ratio > 2.8 and solidity > 0.68:
        conf = round(min(aspect_ratio / 6.0, 1.0), 2)
        return "incision", conf

    # Burn: large area + diffuse/irregular boundary
    if relative_area > 0.10 and circularity < 0.45:
        return "burn", round(1.0 - circularity, 2)

    # Laceration: irregular/jagged edges regardless of size
    if solidity < 0.58:
        conf = round(1.0 - solidity, 2)
        return "laceration", conf

    # Abrasion: moderate-to-large, oval/round, fills much of its bbox
    if relative_area > 0.015 and extent > 0.55 and circularity > 0.30:
        conf = round(min(extent, 1.0), 2)
        return "abrasion", conf

    # Avulsion: fills bbox well but edges aren't smooth
    if extent > 0.55 and solidity < 0.80:
        return "avulsion", round(1.0 - solidity + 0.2, 2)

    # Default fallback
    return "laceration", 0.40


# ---------------------------------------------------------------------------
# Infection risk assessment
# ---------------------------------------------------------------------------

def _assess_infection_risk(
    tissue_type: str,
    tissue_conf: float,
    area_cm2: float | None,
    composition: dict | None = None,
) -> tuple[str, str]:
    """
    Rule-based infection risk flag derived from full tissue composition and
    wound size. Uses the complete composition breakdown (not just the dominant
    type) so that minority tissue components — e.g. 15% necrosis alongside
    predominantly granulation — still raise the risk appropriately.

    Scoring:
      - Any necrosis ≥ 5 %    → +2 per 10 % necrosis (floor at +2)
      - Any slough  ≥ 5 %    → +1 per 10 % slough   (floor at +1)
      - Mixed dominant type   → +1
      - Wound area > 20 cm²  → +1

    Result mapping: score ≥ 3 → high, score ≥ 1 → medium, else → low.

    IMPORTANT LIMITATION: 2D colour analysis cannot assess wound depth.
    A deep cavity (e.g. full-thickness wound) adds significant clinical
    risk not captured here. Always combine with clinical examination.

    Returns (flag, indicators_json).
    """
    indicators: list[str] = []
    score = 0
    comp  = composition or {}

    necro_pct  = comp.get("necrosis",    0.0)
    slough_pct = comp.get("slough",      0.0)
    gran_pct   = comp.get("granulation", 0.0)

    # ── Necrosis contribution ─────────────────────────────────────────────
    # Even small amounts of necrotic tissue are clinically significant.
    if necro_pct >= 0.30:
        score += 3
        indicators.append(
            f"Significant necrotic tissue ({necro_pct*100:.0f}%) — urgent debridement indicated"
        )
    elif necro_pct >= 0.10:
        score += 2
        indicators.append(
            f"Necrotic tissue detected ({necro_pct*100:.0f}%) — debridement recommended"
        )
    elif necro_pct >= 0.05:
        score += 1
        indicators.append(
            f"Minor necrotic areas ({necro_pct*100:.0f}%) — monitor closely"
        )

    # ── Slough / fibrin contribution ──────────────────────────────────────
    if slough_pct >= 0.20:
        score += 2
        indicators.append(
            f"Significant slough/fibrin coating ({slough_pct*100:.0f}%) — "
            "wound bed preparation required"
        )
    elif slough_pct >= 0.05:
        score += 1
        indicators.append(
            f"Slough/fibrin areas detected ({slough_pct*100:.0f}%) — "
            "monitor and consider debridement"
        )

    # ── Dominant-type contribution ────────────────────────────────────────
    if tissue_type == "mixed":
        score += 1
        indicators.append("Mixed tissue composition — complex wound, close monitoring advised")
    elif tissue_type == "granulation" and tissue_conf and tissue_conf >= 0.60:
        indicators.append(
            f"Predominantly granulation tissue ({gran_pct*100:.0f}%) — active healing observed"
        )

    # ── Wound size ────────────────────────────────────────────────────────
    if area_cm2 is not None and area_cm2 > 20.0:
        score += 1
        indicators.append(
            f"Large wound surface area ({area_cm2:.1f} cm²) — increased exposure risk"
        )

    # ── Depth disclaimer (always appended) ───────────────────────────────
    indicators.append(
        "⚠ Wound depth cannot be assessed from 2D photography — "
        "clinical examination required to determine full thickness involvement"
    )

    flag = "high" if score >= 3 else ("medium" if score >= 1 else "low")
    return flag, json.dumps(indicators)


# ---------------------------------------------------------------------------
# cm² estimation helpers
# ---------------------------------------------------------------------------

def _estimate_cm2_from_dpi(image: Image.Image, area_px: float) -> float | None:
    """
    Attempt to infer wound area in cm² from image DPI metadata (EXIF/JFIF).

    Limitation: smartphone cameras almost always embed a generic placeholder
    DPI (72 or 96) unrelated to actual sensor-to-subject distance, so this
    will silently return None for most clinical photos. Reliable cm² requires
    a physical reference object in the frame.
    """
    try:
        dpi_info = image.info.get("dpi")
        if (
            dpi_info
            and isinstance(dpi_info, (tuple, list))
            and len(dpi_info) == 2
        ):
            dpi_x, dpi_y = float(dpi_info[0]), float(dpi_info[1])
            # Reject generic screen/print-placeholder DPIs
            generic = {72.0, 96.0}
            if (
                dpi_x not in generic
                and dpi_y not in generic
                and dpi_x > 0
                and dpi_y > 0
            ):
                cm_per_px_x = 2.54 / dpi_x
                cm_per_px_y = 2.54 / dpi_y
                return area_px * cm_per_px_x * cm_per_px_y
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_report(
    area_cm2: float | None,
    wound_type: str,
    tissue_type: str,
    tissue_confidence: float | None,
    tissue_composition: dict | None,
    infection_risk_flag: str,
    infection_indicators: str,
    trend: str | None,
    change_pct: float | None,
) -> str:
    lines = ["WOUND ASSESSMENT SUMMARY", "-" * 34]

    # Wound type
    lines.append(f"Wound type      : {wound_type.capitalize()}")

    # Area
    if area_cm2:
        lines.append(f"Wound area      : {area_cm2:.2f} cm²")
    else:
        lines.append(
            "Wound area      : not calibrated\n"
            "                  (provide reference object diameter > 0 for cm² measurement)"
        )

    # Tissue type + confidence
    conf_str = (
        f"  ({tissue_confidence * 100:.0f}% confidence)" if tissue_confidence else ""
    )
    lines.append(f"Tissue type     : {tissue_type.capitalize()}{conf_str}")

    # Tissue composition breakdown (if available)
    comp = tissue_composition or {}
    if comp:
        lines.append("  Composition breakdown:")
        for t_name, pct in sorted(comp.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct * 20)
            lines.append(f"    {t_name:<14} {bar:<20} {pct*100:.0f}%")

    # Infection risk
    lines.append(f"Infection risk  : {infection_risk_flag.upper()}")

    # Clinical indicators
    try:
        indicators = json.loads(infection_indicators or "[]")
        for ind in indicators:
            lines.append(f"  • {ind}")
    except Exception:
        pass

    # Healing trend
    if trend and change_pct is not None:
        lines.append(
            f"Healing trend   : {trend.capitalize()} ({change_pct:+.1f}% area change vs. last visit)"
        )
    else:
        lines.append("Healing trend   : baseline visit — no prior visit to compare")

    lines.append("-" * 34)

    # Wound-type specific clinical note
    wound_note_map = {
        "incision":  "NOTE: Clean incised wound — assess depth and consider suturing if edges are approximable.",
        "laceration":"NOTE: Laceration with irregular edges — irrigate thoroughly, consider closure method.",
        "abrasion":  "NOTE: Abrasion wound — clean debris, apply non-adherent dressing, monitor for infection.",
        "burn":      "⚠ BURN WOUND: Assess depth (superficial/partial/full thickness). Refer to burns specialist if deep.",
        "avulsion":  "⚠ AVULSION: Significant tissue loss — specialist review recommended for reconstruction.",
        "puncture":  "NOTE: Puncture wound — high infection risk despite small surface area. Assess depth and tetanus status.",
    }
    wound_note = wound_note_map.get(wound_type)
    if wound_note:
        lines.append(wound_note)

    # Tissue-type specific clinical note
    tissue_note_map = {
        "necrosis":    "⚠ URGENT: Necrotic tissue requires prompt surgical or enzymatic debridement.",
        "slough":      "NOTE: Wound bed preparation and debridement recommended before re-dressing.",
        "granulation": "NOTE: Granulation tissue indicates active healing. Maintain moist wound environment.",
        "mixed":       "NOTE: Multiple tissue types present. Reassess after next dressing change.",
    }
    tissue_note = tissue_note_map.get(tissue_type)
    if tissue_note:
        lines.append(tissue_note)

    return "\n".join(lines)

