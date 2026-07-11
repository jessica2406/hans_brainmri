"""
central_location.py (v2 - fixed per PR review)

Independent (non-ResNet) predicate extractor for HAN-S.

Fix vs v1: the original version could fall back to the largest
contour in the image, which on scans with no distinct lesion (e.g.
no_tumor) or a subtle one is often the skull/whole-brain outline.
Since that outline is roughly centered by definition, the predicate
was effectively meaningless (near-constant True regardless of true
class). This version explicitly excludes skull-sized / border-
touching contours before computing a centroid, so central_location
only fires when a genuine, spatially distinct lesion is found near
the image center.

Usage:
    python central_location.py path/to/mri.png
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


def extract_central_location(
    image_path,
    threshold_value=160,
    distance_ratio_threshold=0.20,
    min_area_fraction=0.005,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Threshold + contour detection.
      2. EXCLUDE skull/whole-brain contours (large and/or
         border-touching) so we don't centroid the whole head.
      3. Among remaining candidates, pick the largest as the lesion.
      4. Compute normalized centroid distance from image center.
      5. central_location = True only if a genuine (non-skull) lesion
         candidate exists AND its centroid is close to center.
         If no valid lesion candidate is found, the predicate is
         False (not a fallback True) - "no lesion" should never be
         confused with "centrally located lesion".
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    image_center = (w / 2.0, h / 2.0)
    diagonal = np.sqrt(w ** 2 + h ** 2)
    img_area = h * w

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = [
        c for c in contours
        if cv2.contourArea(c) > min_area_fraction * img_area
        and not _is_probably_skull(c, (h, w))
    ]

    if not candidates:
        # No genuine, spatially distinct lesion candidate found.
        # This must NOT default to True - absence of a lesion is not
        # evidence of a centrally-located one.
        return False, {
            "reason": "no non-skull lesion candidate found",
            "centroid": None,
            "normalized_distance": None,
            "debug_image": None,
        }

    lesion = max(candidates, key=cv2.contourArea)
    M = cv2.moments(lesion)

    if M["m00"] == 0:
        return False, {
            "reason": "degenerate contour (zero area moment)",
            "centroid": None,
            "normalized_distance": None,
            "debug_image": None,
        }

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    distance = np.sqrt((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2)
    normalized_distance = distance / diagonal

    predicate = normalized_distance < distance_ratio_threshold

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [lesion], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(image_center[0]), int(image_center[1])), 5, (255, 0, 0), -1)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_central_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "centroid": (round(cx, 1), round(cy, 1)),
        "normalized_distance": round(normalized_distance, 3),
        "threshold_used": distance_ratio_threshold,
        "num_candidates_considered": len(candidates),
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
    print(f"  normalized_distance: {details.get('normalized_distance')}")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")