"""
Wound segmentation model wrapper -- MedSAM implementation.

Problems found during testing, in order, and how each was fixed:

1. A single full-image box prompt on a mostly-black padded photo just
   returns the box itself. Fix: crop to actual photo content first
   (non-black region).
2. On small crops, SAM's automatic mask generator (grid-of-points
   candidate proposal) came back empty -- unreliable at small sizes.
   Fix: switched to a targeted box prompt via SamPredictor instead.
3. On a real, unpadded photo where the wound is a small part of a much
   larger visible skin/arm area, a box prompt covering the whole photo
   makes SAM segment the *dominant object in the scene* (the arm), not
   the wound sitting inside it -- SAM has no way to know "wound" is what
   we care about from geometry alone.
4. The initial HSV red-thresholding was too permissive (low saturation
   cutoff) and was catching the surrounding skin (which is pinkish/beige)
   as well as the wound. Fix: raised the saturation threshold and rank
   candidate components by mean saturation (not raw area) so we pick the
   most vividly-coloured region, which is the actual wound tissue.

Current approach:
  a. Crop to actual photo content (strips black padding, if present).
  b. Localize the wound region using color: blood/open wound tissue is
     distinctly more saturated red than surrounding skin, so threshold
     in HSV space (high saturation red only), then rank connected
     components by mean saturation inside the component and take the
     best one's bounding box. This is the prompt-generation step --
     it decides *where* to point SAM.
  c. Run SamPredictor with a box prompt around that localized region
     (not the whole photo), returning 3 candidate masks, and pick the
     best via SAM's own predicted quality score.
  d. Paste the resulting mask back into full-image coordinates.

Known limitation: the color-based localization step assumes the wound
is redder/more saturated than the surrounding skin. This holds for
fresh/bleeding wounds and granulation tissue but will likely fail on:
  - Necrotic tissue (black/brown)
  - Heavy slough coverage (yellow/white)
  - Unusual/artificial lighting that shifts hues
It is a placeholder for a trained wound-localization step, not a robust
general solution -- revisit once you have labeled data.

Checkpoint: download via ml/download_medsam_checkpoint.py first, and set
MEDSAM_CHECKPOINT_PATH in backend/.env to point at it.
"""

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_TYPE = "vit_b"  # MedSAM's public checkpoint is fine-tuned from SAM ViT-B

CONTENT_DARK_THRESHOLD = 15   # pixel considered "padding" if max(R,G,B) below this
CROP_MARGIN_PX = 4            # small margin kept around detected content bbox
LOCALIZATION_MARGIN_FRAC = 0.10  # margin added around the wound bbox for the SAM box prompt

# HSV red ranges -- only high-saturation reds to avoid matching pale skin.
# Saturation floor raised to 100/255 (~39 %) so beige/pink skin (S≈30-70) is
# excluded while vivid wound tissue (S≈120-255) is kept.
# Value floor raised to 50 to exclude very dark pixels (shadows, hair).
HSV_RED_RANGES = [
    ((0,  100, 50), (12, 255, 255)),   # warm reds / granulation tissue
    ((165, 100, 50), (179, 255, 255)), # deep reds wrapping the hue wheel
]

# Ignore tiny noise components below this fraction of the cropped image area.
MIN_COMPONENT_AREA_FRAC = 0.002

# Fallback box used when NO red region is found: tight center crop (40 % of
# each axis), not near-full-frame.  A tighter box gives SAM a much better
# chance of picking the wound rather than the whole limb.
FALLBACK_BOX_INNER = 0.30   # start at 30 % from each edge
FALLBACK_BOX_OUTER = 0.70   # end  at 70 % from each edge


class WoundSegmenter:
    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        if checkpoint_path:
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[MODEL_TYPE](checkpoint=None)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        sam.load_state_dict(state_dict)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Returns a binary mask (H, W) with 1 = wound pixel, 0 = background,
        sized to match the original (uncropped) input image.
        """
        if self.predictor is None:
            return self._stub_segment(image)

        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]

        # Step a: strip black padding
        x0, y0, x1, y1 = self._content_bbox(image_np)
        cropped = image_np[y0:y1, x0:x1]
        ch, cw = cropped.shape[:2]
        if ch == 0 or cw == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # Step b: find the wound box via color
        box = self._localize_wound_box(cropped)

        # Step c: SAM box-prompt prediction on the cropped region
        self.predictor.set_image(cropped)
        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        crop_mask = masks[best_idx].astype(np.uint8)

        # Step d: paste back into full-image coordinates
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = crop_mask
        return full_mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_bbox(image_np: np.ndarray) -> tuple[int, int, int, int]:
        """
        Bounding box of non-black (non-padding) content.
        Falls back to the full image if no dark border is detected.
        """
        h, w = image_np.shape[:2]
        content = image_np.max(axis=2) > CONTENT_DARK_THRESHOLD

        rows = np.where(content.any(axis=1))[0]
        cols = np.where(content.any(axis=0))[0]

        if rows.size == 0 or cols.size == 0:
            return 0, 0, w, h

        y0 = max(int(rows.min()) - CROP_MARGIN_PX, 0)
        y1 = min(int(rows.max()) + CROP_MARGIN_PX + 1, h)
        x0 = max(int(cols.min()) - CROP_MARGIN_PX, 0)
        x1 = min(int(cols.max()) + CROP_MARGIN_PX + 1, w)
        return x0, y0, x1, y1

    @staticmethod
    def _localize_wound_box(image_np: np.ndarray) -> np.ndarray:
        """
        Finds the bounding box of the most wound-like (vivid red / saturated)
        connected region and returns it as [x0, y0, x1, y1] for SamPredictor.

        Scoring ranks components by their *mean HSV saturation* inside the
        mask (not raw area) so that a smaller but more vividly-coloured wound
        beats a larger, paler skin region.
        """
        h, w = image_np.shape[:2]
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        # Build red-pixel mask
        red_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in HSV_RED_RANGES:
            red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Clean up noise with morphological ops
        kernel = np.ones((7, 7), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            red_mask, connectivity=8
        )

        total_px = h * w
        best_label = None
        best_score = -1.0

        saturation_channel = hsv[:, :, 1].astype(np.float32)  # 0..255

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area / total_px < MIN_COMPONENT_AREA_FRAC:
                continue  # skip tiny specks

            # Score = mean saturation of pixels inside this component.
            # Wound tissue is vivid (high S); skin is muted (low S).
            component_pixels = saturation_channel[labels == label]
            mean_sat = float(component_pixels.mean())

            if mean_sat > best_score:
                best_score = mean_sat
                best_label = label

        # No qualifying red region found -- use a tight center fallback box.
        if best_label is None:
            return np.array([
                int(w * FALLBACK_BOX_INNER),
                int(h * FALLBACK_BOX_INNER),
                int(w * FALLBACK_BOX_OUTER),
                int(h * FALLBACK_BOX_OUTER),
            ])

        # Build the box from the best component's bounding rect + margin.
        bx = stats[best_label, cv2.CC_STAT_LEFT]
        by = stats[best_label, cv2.CC_STAT_TOP]
        bw = stats[best_label, cv2.CC_STAT_WIDTH]
        bh = stats[best_label, cv2.CC_STAT_HEIGHT]

        mx = int(bw * LOCALIZATION_MARGIN_FRAC)
        my = int(bh * LOCALIZATION_MARGIN_FRAC)
        x0 = max(bx - mx, 0)
        y0 = max(by - my, 0)
        x1 = min(bx + bw + mx, w)
        y1 = min(by + bh + my, h)
        return np.array([x0, y0, x1, y1])

    @staticmethod
    def _stub_segment(image: Image.Image) -> np.ndarray:
        """Fallback used when no MedSAM checkpoint is loaded."""
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        cx0, cy0 = int(w * 0.3), int(h * 0.3)
        cx1, cy1 = int(w * 0.7), int(h * 0.7)
        mask[cy0:cy1, cx0:cx1] = 1
        return mask