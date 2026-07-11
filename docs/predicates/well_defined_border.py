"""
well_defined_border.py (v3 - recalibrated based on real test data)

Independent (non-ResNet) predicate extractor for HAN-S.

Fix vs v2: raw Canny edge strength around the lesion contour turned
out to be a poor discriminator in practice - real testing across
meningioma, no_tumor, pituitary, and glioma samples showed edge
strength clustered tightly (48-63) regardless of class, giving a
near-constant False.

Solidity (Area / ConvexHullArea) showed much better separation on the
same test data: meningioma ~0.93 (smooth, convex, well-defined) vs
no_tumor/pituitary/glioma ~0.68-0.73 (more irregular/less convex).
This matches the clinical rationale directly: a well-defined border
is smooth and convex, which solidity measures more directly than
average edge brightness does. This version uses solidity (plus a
circularity sanity check) as the primary signal, calibrated against
the midpoint between the observed clusters.

NOTE: calibrated on a small sample (1 image per class). Thresholds
should be revisited once more images per class are tested.

Usage:
    python well_defined_border.py path/to/mri.png
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


def extract_well_defined_border(
    image_path,
    solidity_threshold=0.85,
    min_area_fraction=0.005,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Otsu-thresholded binary mask.
      2. Exclude skull/whole-brain contours.
      3. Solidity = Area / ConvexHullArea. Smooth, well-bounded
         lesions (e.g. meningioma) are highly convex (~0.9+);
         irregular/infiltrative or diffuse lesions score lower.
      4. well_defined_border = True if solidity exceeds threshold.
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