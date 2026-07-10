"""
irregular_border.py

Independent (non-ResNet) predicate extractor for HAN-S.

Estimates whether a lesion in a brain MRI slice has an irregular,
infiltrative border using classical shape analysis (circularity,
solidity, convex hull), as proposed in
docs/Independent_AIL_Predicates.md.

Usage:
    python irregular_border.py path/to/mri.png

Output:
    Prints irregular_border = True/False
    Saves a debug image "<input>_irregular_debug.png" showing the
    lesion contour and its convex hull.
"""

import sys
import os
import cv2
import numpy as np


def extract_irregular_border(
    image_path,
    threshold_value=160,
    circularity_threshold=0.65,
    solidity_threshold=0.85,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Threshold + contour detection to isolate the candidate lesion
         (largest contour above a minimum area).
      2. Circularity = 4*pi*Area / Perimeter^2
         - A perfect circle scores 1.0; irregular/infiltrative shapes
           score noticeably lower.
      3. Solidity = Area / ConvexHullArea
         - A convex, well-bounded shape scores close to 1.0; a shape
           with concavities/spiculations scores lower.
      4. irregular_border = True if circularity is low AND solidity is
         low (i.e. the shape is both non-circular and non-convex).
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
    min_area = 0.01 * img_area

    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        predicate = False
        details = {
            "reason": "no contour of sufficient size found",
            "circularity": None,
            "solidity": None,
            "debug_image": None,
        }
        return predicate, details

    lesion = max(valid_contours, key=cv2.contourArea)

    area = cv2.contourArea(lesion)
    perimeter = cv2.arcLength(lesion, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    hull = cv2.convexHull(lesion)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    predicate = (
        circularity < circularity_threshold
        and solidity < solidity_threshold
    )

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
        "circularity_threshold": circularity_threshold,
        "solidity_threshold": solidity_threshold,
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