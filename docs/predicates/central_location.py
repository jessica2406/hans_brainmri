"""
central_location.py

Independent (non-ResNet) predicate extractor for HAN-S.

Estimates whether a lesion in a brain MRI slice is centrally located
(consistent with a sellar/pituitary mass) using classical centroid
analysis, as proposed in docs/Independent_AIL_Predicates.md.

Usage:
    python central_location.py path/to/mri.png

Output:
    Prints central_location = True/False
    Saves a debug image "<input>_central_debug.png" showing the lesion
    centroid, the image center, and the distance between them.
"""

import sys
import os
import cv2
import numpy as np


def extract_central_location(
    image_path,
    threshold_value=160,
    distance_ratio_threshold=0.25,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Threshold + contour detection to isolate the candidate lesion
         (largest contour above a minimum area).
      2. Compute the lesion's centroid via image moments.
      3. Compute the image's geometric center.
      4. Normalize the Euclidean distance between the two by the image
         diagonal, so the threshold is resolution-independent.
      5. central_location = True if the normalized distance is below
         distance_ratio_threshold.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    image_center = (w / 2.0, h / 2.0)
    diagonal = np.sqrt(w ** 2 + h ** 2)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, threshold_value, 255, cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_area = h * w
    min_area = 0.01 * img_area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        predicate = False
        details = {
            "reason": "no contour of sufficient size found",
            "centroid": None,
            "image_center": image_center,
            "normalized_distance": None,
            "debug_image": None,
        }
        return predicate, details

    lesion = max(valid_contours, key=cv2.contourArea)
    M = cv2.moments(lesion)

    if M["m00"] == 0:
        predicate = False
        details = {
            "reason": "degenerate contour (zero area moment)",
            "centroid": None,
            "image_center": image_center,
            "normalized_distance": None,
            "debug_image": None,
        }
        return predicate, details

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    distance = np.sqrt((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2)
    normalized_distance = distance / diagonal

    predicate = normalized_distance < distance_ratio_threshold

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [lesion], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # lesion centroid
        cv2.circle(debug_img, (int(image_center[0]), int(image_center[1])), 5, (255, 0, 0), -1)  # image center
        cv2.line(
            debug_img,
            (int(cx), int(cy)),
            (int(image_center[0]), int(image_center[1])),
            (0, 255, 255),
            1,
        )
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_central_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "centroid": (round(cx, 1), round(cy, 1)),
        "image_center": (round(image_center[0], 1), round(image_center[1], 1)),
        "distance_px": round(distance, 1),
        "normalized_distance": round(normalized_distance, 3),
        "threshold_used": distance_ratio_threshold,
        "debug_image": debug_path,
    }

    return predicate, details


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python central_location.py path/to/mri.png")
        sys.exit(1)

    image_path = sys.argv[1]
    predicate, details = extract_central_location(image_path)

    print(f"central_location = {predicate}")
    print(f"  centroid: {details.get('centroid')}")
    print(f"  image center: {details.get('image_center')}")
    print(f"  normalized distance: {details.get('normalized_distance')}")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")