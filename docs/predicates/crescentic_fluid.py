"""
crescentic_fluid.py

Independent (non-ResNet) predicate extractor for HAN-S.

Detects crescent-shaped fluid collections in a brain MRI slice using
classical computer vision (no neural network involved), as proposed in
docs/Independent_AIL_Predicates.md.

Usage:
    python crescentic_fluid.py path/to/mri.png

Output:
    Prints crescentic_fluid = True/False
    Saves a debug image "<input>_crescentic_debug.png" showing the
    contour(s) that were considered.
"""

import sys
import os
import cv2
import numpy as np


def extract_crescentic_fluid(image_path, threshold_value=180, debug=True):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Gaussian blur to reduce noise.
      2. Binary threshold to isolate bright fluid-intensity regions.
      3. Contour detection on the thresholded mask.
      4. Shape analysis: a crescent is characterised by
         - moderate-to-low solidity (concave shape, unlike a solid blob)
         - an elongated, curved footprint (high perimeter^2 / area ratio)
         - non-trivial area (filters out noise specks)
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, threshold_value, 255, cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_area = gray.shape[0] * gray.shape[1]
    min_area = 0.002 * img_area  # ignore tiny noise contours

    best_candidate = None
    best_score = -1
    candidates_checked = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Crescents are concave (solidity noticeably < 1) and elongated
        # (high perimeter^2/area relative to a circle, where 4*pi ~= 12.57)
        shape_factor = (perimeter ** 2) / area if area > 0 else 0

        is_concave = 0.4 < solidity < 0.9
        is_elongated = shape_factor > 20  # circles ~12.6, crescents higher

        candidates_checked.append({
            "area": area,
            "perimeter": perimeter,
            "solidity": round(solidity, 3),
            "shape_factor": round(shape_factor, 2),
            "is_concave": is_concave,
            "is_elongated": is_elongated,
        })

        if is_concave and is_elongated:
            score = (1 - solidity) * shape_factor
            if score > best_score:
                best_score = score
                best_candidate = cnt

    predicate = best_candidate is not None

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 1)
        if best_candidate is not None:
            cv2.drawContours(debug_img, [best_candidate], -1, (0, 255, 0), 2)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_crescentic_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "threshold_value": threshold_value,
        "num_contours_found": len(contours),
        "num_candidates_checked": len(candidates_checked),
        "candidates": candidates_checked,
        "debug_image": debug_path,
    }

    return predicate, details


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crescentic_fluid.py path/to/mri.png")
        sys.exit(1)

    image_path = sys.argv[1]
    predicate, details = extract_crescentic_fluid(image_path)

    print(f"crescentic_fluid = {predicate}")
    print(f"  contours found:   {details['num_contours_found']}")
    print(f"  candidates checked: {details['num_candidates_checked']}")
    if details["debug_image"]:
        print(f"  debug image saved: {details['debug_image']}")