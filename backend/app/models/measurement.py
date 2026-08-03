import numpy as np


def mask_area_px(mask: np.ndarray) -> float:
    """Count of wound pixels in a binary mask."""
    return float(mask.sum())


def px_to_cm2(area_px: float, reference_object_diameter_cm: float, reference_object_diameter_px: float) -> float:
    """
    Converts a pixel area to cm^2 using a known reference object (e.g. a coin or
    calibration sticker) whose real-world diameter and pixel diameter are both known.

    cm_per_px = reference_object_diameter_cm / reference_object_diameter_px
    area_cm2 = area_px * (cm_per_px ** 2)
    """
    if reference_object_diameter_px <= 0:
        raise ValueError("reference_object_diameter_px must be > 0")
    cm_per_px = reference_object_diameter_cm / reference_object_diameter_px
    return area_px * (cm_per_px ** 2)


def area_change_pct(current_cm2: float, previous_cm2: float) -> float:
    """Positive = wound grew, negative = wound shrank (healing)."""
    if previous_cm2 == 0:
        return 0.0
    return ((current_cm2 - previous_cm2) / previous_cm2) * 100.0


def classify_trend(change_pct: float) -> str:
    if change_pct <= -10:
        return "improving"
    elif change_pct >= 10:
        return "worsening"
    return "stable"
