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
   the wound sitting inside it.
4. The initial HSV red-thresholding was too permissive and was catching
   surrounding skin. Fix: raised the saturation threshold and rank
   candidate components by mean saturation (not area).
5. SAM's own quality score sometimes assigns higher confidence to a large
   skin/background mask than to the smaller wound mask. Fix: switch from
   score-based selection to overlap-based selection — pick the SAM mask
   that overlaps best with our initial red localization region.
6. No foreground-point hint was given to SAM. Fix: pass the centroid of
   the best red component as a foreground point in addition to the box
   prompt, and add image-corner background points so SAM knows the
   periphery is NOT the wound.

Current approach:
  a. Crop to actual photo content (strips black padding, if present).
  b. Localize the wound using HSV: rank connected components by mean
     saturation; take the best component's bounding box and centroid.
  c. Run SamPredictor with a box prompt + foreground centroid point +
     background corner points (multimask_output=True for 3 candidates).
  d. Select the best candidate mask by IoU with the initial red region
     (not SAM's internal quality score, which can prefer large regions).
  e. Paste the mask back into full-image coordinates.

Known limitation: assumes wound is redder / more saturated than skin.
Works well for fresh wounds and granulation tissue; may fail on:
  - Necrotic tissue (brown/black)
  - Heavy slough (yellow/white)
  - Unusual lighting
Revisit once labeled training data is available.
"""

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_TYPE = "vit_b"  # MedSAM's public checkpoint is fine-tuned from SAM ViT-B

CONTENT_DARK_THRESHOLD = 15   # pixel considered "padding" if max(R,G,B) below this
CROP_MARGIN_PX = 4            # small margin kept around detected content bbox
LOCALIZATION_MARGIN_FRAC = 0.08  # margin added around wound bbox for SAM prompt (tightened)

# HSV red ranges — only high-saturation reds to avoid matching pale skin.
# Saturation floor = 110/255 (~43 %) so pink skin (S≈30-80) is excluded
# while vivid wound tissue (S≈120-255) is kept.
# Value floor = 50 to exclude very dark pixels (shadows, hair, necrosis).
HSV_RED_RANGES = [
    ((0,  110, 50), (12, 255, 255)),    # warm reds / granulation tissue
    ((165, 110, 50), (179, 255, 255)),  # deep reds wrapping the hue wheel
]

# Ignore tiny noise components below this fraction of the cropped image area.
MIN_COMPONENT_AREA_FRAC = 0.002

# Fallback box when NO red region found: tight center crop (35–65 %).
FALLBACK_BOX_INNER = 0.35
FALLBACK_BOX_OUTER = 0.65


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

        # Step b: localise wound — returns box, centroid, and approximate mask
        box, fg_point, ref_mask = self._localize_wound(cropped)

        # Step c: build SAM prompts
        #   - foreground point: centroid of the best red component (tells SAM
        #     "the wound is HERE, specifically at this pixel")
        #   - background points: image corners (tells SAM "the periphery is
        #     NOT the wound") — crucial for images where skin fills the frame
        corner_frac = 0.05
        ch_c, cw_c = cropped.shape[:2]
        bg_points = np.array([
            [int(cw_c * corner_frac),       int(ch_c * corner_frac)],        # top-left
            [int(cw_c * (1 - corner_frac)), int(ch_c * corner_frac)],        # top-right
            [int(cw_c * corner_frac),       int(ch_c * (1 - corner_frac))],  # bottom-left
            [int(cw_c * (1 - corner_frac)), int(ch_c * (1 - corner_frac))],  # bottom-right
        ])

        if fg_point is not None:
            point_coords  = np.vstack([fg_point, bg_points])
            point_labels  = np.array([1, 0, 0, 0, 0])   # 1 = foreground, 0 = background
        else:
            point_coords = bg_points
            point_labels = np.zeros(len(bg_points), dtype=int)

        # Step d: SAM prediction
        self.predictor.set_image(cropped)
        masks, scores, _ = self.predictor.predict(
            box=box,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Step e: select best mask by IoU with our red reference region
        #   (NOT by SAM's quality score — scores can prefer large skin masks)
        best_idx = self._select_best_mask(masks, ref_mask, scores)
        crop_mask = masks[best_idx].astype(np.uint8)

        # Step f: paste back into full-image coordinates
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = crop_mask
        return full_mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_bbox(image_np: np.ndarray) -> tuple[int, int, int, int]:
        """Bounding box of non-black (non-padding) content."""
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
    def _localize_wound(
        image_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Finds the wound region using HSV color thresholding.

        Returns:
          box       [x0, y0, x1, y1] SAM box prompt
          fg_point  [[cx, cy]] centroid of the best component (or None)
          ref_mask  binary mask of the best red component (for overlap scoring)
        """
        h, w = image_np.shape[:2]
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)

        # Build red-pixel mask
        red_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in HSV_RED_RANGES:
            red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Clean up noise
        kernel = np.ones((7, 7), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            red_mask, connectivity=8
        )

        total_px   = h * w
        best_label = None
        best_score = -1.0

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area / total_px < MIN_COMPONENT_AREA_FRAC:
                continue   # skip tiny specks
            # Rank by mean saturation (wound > skin)
            mean_sat = float(saturation[labels == label].mean())
            if mean_sat > best_score:
                best_score = mean_sat
                best_label = label

        # No qualifying red region — use tight center fallback
        if best_label is None:
            box = np.array([
                int(w * FALLBACK_BOX_INNER),
                int(h * FALLBACK_BOX_INNER),
                int(w * FALLBACK_BOX_OUTER),
                int(h * FALLBACK_BOX_OUTER),
            ])
            ref_mask = np.zeros((h, w), dtype=np.uint8)
            return box, None, ref_mask

        # Build a tight bounding box + small margin
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
        box = np.array([x0, y0, x1, y1])

        # Foreground centroid point
        cx, cy = centroids[best_label]
        fg_point = np.array([[cx, cy]])

        # Component mask used for overlap-based selection
        ref_mask = (labels == best_label).astype(np.uint8)

        return box, fg_point, ref_mask

    @staticmethod
    def _select_best_mask(
        masks: np.ndarray,
        ref_mask: np.ndarray,
        scores: np.ndarray,
    ) -> int:
        """
        Choose the SAM candidate mask that best overlaps our HSV reference
        region. Ties or zero-reference fall back to SAM's own quality score.

        Using IoU with the initial red-pixel region ensures we pick the wound
        mask even when SAM gives a higher confidence score to a larger
        background or skin-region mask.
        """
        if ref_mask.sum() == 0:
            return int(np.argmax(scores))

        best_idx = 0
        best_iou = -1.0

        for i, mask in enumerate(masks):
            m     = mask.astype(bool)
            r     = ref_mask.astype(bool)
            inter = (m & r).sum()
            union = (m | r).sum()
            iou   = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return best_idx

    @staticmethod
    def _stub_segment(image: Image.Image) -> np.ndarray:
        """
        Fallback used when no MedSAM checkpoint is loaded.
        Uses HSV color thresholding directly to produce a rough mask —
        better than a fixed rectangle which ignores where the wound actually is.
        """
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in HSV_RED_RANGES:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Morphological cleanup
        kernel = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Fill holes inside the wound region
        mask_filled = mask.copy()
        flood_fill  = mask.copy()
        flood_mask  = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood_fill, flood_mask, (0, 0), 255)
        mask_filled = mask | cv2.bitwise_not(flood_fill)

        return (mask_filled > 0).astype(np.uint8)