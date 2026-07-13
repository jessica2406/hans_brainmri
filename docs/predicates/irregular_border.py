"""
irregular_border.py (v2 - fixed, consistent with well_defined_border/central_location fixes)

Independent (non-ResNet) predicate extractor for HAN-S.

Fix vs v1:
  1. Used a fixed intensity threshold (160), which failed to find any
     contour at all on some real images (returned None/None). Switched
     to Otsu's method so the threshold adapts per-image.
  2. Did not exclude the skull/whole-brain contour, so on images where
     the skull was the largest contour, circularity/solidity were
     being computed on the wrong shape entirely. Added the same
     skull-exclusion heuristic used in well_defined_border.py and
     central_location.py for consistency across all predicates.

Usage:
    python irregular_border.py path/to/mri.png
"""

import sys
import os
import cv2
import numpy as np


def _is_probably_skull(cnt, img_shape, area_fraction_limit=0.35, border_margin=5):
    h, w = img_shape
    img_area = h * w
    area = cv2.contourArea(cnt)
    if area > area_fraction_limit * img_area:
        return True

    x, y, cw, ch = cv2.boundingRect(cnt)
    touches_border = (
        x <= border_margin
        or y <= border_margin
        or (x + cw) >= (w - border_margin)
        or (y + ch) >= (h - border_margin)
    )
    return touches_border


def extract_irregular_border(
    image_path,
    circularity_threshold=0.65,
    solidity_threshold=0.85,
    min_area_fraction=0.005,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Otsu-thresholded binary mask (adapts per image, unlike a
         fixed intensity cutoff).
      2. Exclude skull/whole-brain contours.
      3. Circularity = 4*pi*Area / Perimeter^2 (1.0 = perfect circle).
      4. Solidity = Area / ConvexHullArea (1.0 = fully convex).
      5. irregular_border = True if both circularity and solidity are
         low (non-circular AND non-convex).
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    img_area = h * w

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = [
        c for c in contours
        if cv2.contourArea(c) > min_area_fraction * img_area
        and not _is_probably_skull(c, (h, w))
    ]

    if not candidates:
        return False, {
            "reason": "no non-skull lesion candidate found",
            "circularity": None,
            "solidity": None,
            "debug_image": None,
        }

    lesion = max(candidates, key=cv2.contourArea)

    area = cv2.contourArea(lesion)
    perimeter = cv2.arcLength(lesion, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    hull = cv2.convexHull(lesion)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    predicate = circularity < circularity_threshold and solidity < solidity_threshold

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [lesion], -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, [hull], -1, (255, 0, 0), 1)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_irregular_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "area": area,
        "perimeter": round(perimeter, 2),
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "num_candidates_considered": len(candidates),
        "debug_image": debug_path,
    }

    return predicate, details


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python irregular_border.py path/to/mri.png")
        sys.exit(1)

    image_path = sys.argv[1]
    predicate, details = extract_irregular_border(image_path)

    print(f"irregular_border = {predicate}")
    print(f"  circularity: {details.get('circularity')}")
    print(f"  solidity: {details.get('solidity')}")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")