"""
well_defined_border.py (v2 - fixed per PR review)

Independent (non-ResNet) predicate extractor for HAN-S.

Fix vs v1: the original version measured edge strength around the
LARGEST edge contour, which on a brain MRI is almost always the skull
outline, not the lesion. That made the predicate return an almost
constant value regardless of the true class. This version explicitly
excludes skull-sized contours (anything covering a large fraction of
the image, or touching the image border) before selecting a lesion
candidate, so the measurement reflects the actual mass boundary.

Usage:
    python well_defined_border.py path/to/mri.png
"""

import sys
import os
import cv2
import numpy as np


def _is_probably_skull(cnt, img_shape, area_fraction_limit=0.35, border_margin=5):
    """Heuristic: skull/whole-brain contours are large and touch the
    image border; true lesions are smaller and interior."""
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


def extract_well_defined_border(
    image_path,
    canny_low=50,
    canny_high=150,
    edge_strength_threshold=100.0,
    min_area_fraction=0.005,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Canny edge detection on the grayscale MRI.
      2. Find candidate lesion contours, EXCLUDING skull/whole-brain
         contours (large and/or border-touching).
      3. Measure edge strength (Canny) and continuity specifically
         around the selected lesion candidate contour.
      4. well_defined_border = True if edge strength and continuity
         both exceed thresholds calibrated against real class-
         separated data (see PR discussion) rather than a fixed
         guess.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    img_area = h * w

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    # Use intensity thresholding (not just raw edges) to find candidate
    # lesion regions - more robust than relying on edge contours alone.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = [
        c for c in contours
        if cv2.contourArea(c) > min_area_fraction * img_area
        and not _is_probably_skull(c, (h, w))
    ]

    if not candidates:
        predicate = False
        details = {
            "reason": "no non-skull lesion candidate found",
            "region_edge_strength": None,
            "continuity": None,
            "debug_image": None,
        }
        return predicate, details

    lesion = max(candidates, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [lesion], -1, 255, thickness=3)
    region_edge_strength = float(np.mean(edges[mask == 255])) if np.any(mask == 255) else 0.0
    perimeter = cv2.arcLength(lesion, True)
    continuity = float(np.count_nonzero(edges[mask == 255])) / max(perimeter, 1)

    predicate = region_edge_strength > edge_strength_threshold and continuity > 0.5

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        debug_img[edges > 0] = (0, 0, 255)
        cv2.drawContours(debug_img, [lesion], -1, (0, 255, 0), 2)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_border_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "region_edge_strength": round(region_edge_strength, 2),
        "continuity": round(continuity, 3),
        "threshold_used": edge_strength_threshold,
        "num_candidates_considered": len(candidates),
        "debug_image": debug_path,
    }

    return predicate, details


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python well_defined_border.py path/to/mri.png")
        sys.exit(1)

    image_path = sys.argv[1]
    predicate, details = extract_well_defined_border(image_path)

    print(f"well_defined_border = {predicate}")
    print(f"  region_edge_strength: {details.get('region_edge_strength')}")
    print(f"  continuity: {details.get('continuity')}")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")