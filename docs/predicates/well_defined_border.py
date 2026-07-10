"""
well_defined_border.py

Independent (non-ResNet) predicate extractor for HAN-S.

Estimates whether a lesion in a brain MRI slice has a smooth, sharply
defined border using classical edge detection, as proposed in
docs/Independent_AIL_Predicates.md.

Usage:
    python well_defined_border.py path/to/mri.png

Output:
    Prints well_defined_border = True/False
    Saves a debug image "<input>_border_debug.png" showing detected edges.
"""

import sys
import os
import cv2
import numpy as np


def extract_well_defined_border(
    image_path,
    canny_low=50,
    canny_high=150,
    edge_strength_threshold=15.0,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Canny edge detection on the grayscale MRI.
      2. Sobel gradient magnitude as a secondary sharpness signal.
      3. Average edge strength = mean of the Canny edge map restricted
         to the region around the largest detected contour (the
         candidate lesion), rather than the whole image, so background
         noise doesn't dilute the score.
      4. well_defined_border = True if edge strength exceeds threshold
         AND edges form a mostly continuous (non-fragmented) boundary.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blur, canny_low, canny_high)

    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)

    # Find largest contour on the edge map to localize the lesion boundary
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, thickness=3)
        region_edge_strength = float(np.mean(edges[mask == 255])) if np.any(mask == 255) else 0.0
        region_sobel_strength = float(np.mean(sobel_mag[mask == 255])) if np.any(mask == 255) else 0.0
        perimeter = cv2.arcLength(largest, True)
        # Continuity: ratio of actual edge pixels on the contour path vs
        # expected pixel count if boundary were fully traced
        continuity = float(np.count_nonzero(edges[mask == 255])) / max(perimeter, 1)
    else:
        largest = None
        region_edge_strength = 0.0
        region_sobel_strength = 0.0
        continuity = 0.0

    predicate = (
        region_edge_strength > edge_strength_threshold
        and continuity > 0.5
    )

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        debug_img[edges > 0] = (0, 0, 255)
        if largest is not None:
            cv2.drawContours(debug_img, [largest], -1, (0, 255, 0), 2)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_border_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "canny_low": canny_low,
        "canny_high": canny_high,
        "region_edge_strength": round(region_edge_strength, 2),
        "region_sobel_strength": round(region_sobel_strength, 2),
        "continuity": round(continuity, 3),
        "threshold_used": edge_strength_threshold,
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
    print(f"  edge strength (region): {details['region_edge_strength']}")
    print(f"  sobel strength (region): {details['region_sobel_strength']}")
    print(f"  continuity: {details['continuity']}")
    if details["debug_image"]:
        print(f"  debug image saved: {details['debug_image']}")