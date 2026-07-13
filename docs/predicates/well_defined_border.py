"""
well_defined_border.py (v4 - uses shared, tightened segmentation)

Independent (non-ResNet) predicate extractor for HAN-S.

See lesion_segmentation.py for the segmentation fix details (adds
bounding-box coverage, extent, and aspect-ratio checks on top of the
previous area/border-touch filter, after debug-image inspection showed
a near-whole-brain-outline contour was slipping through the old
filter on at least one image).

Usage:
    python well_defined_border.py path/to/mri.png
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesion_segmentation import get_lesion_candidates


def extract_well_defined_border(
    image_path,
    solidity_threshold=0.85,
    min_area_fraction=0.005,
    debug=True,
):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = get_lesion_candidates(gray, contours, min_area_fraction)

    if not candidates:
        return False, {
            "reason": "no lesion candidate found after segmentation filtering",
            "solidity": None,
            "debug_image": None,
        }

    lesion = max(candidates, key=cv2.contourArea)

    area = cv2.contourArea(lesion)
    hull = cv2.convexHull(lesion)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    predicate = solidity > solidity_threshold

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [lesion], -1, (0, 255, 0), 2)
        cv2.drawContours(debug_img, [hull], -1, (255, 0, 0), 1)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_border_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "solidity": round(solidity, 3),
        "threshold_used": solidity_threshold,
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
    print(f"  solidity: {details.get('solidity')}")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")