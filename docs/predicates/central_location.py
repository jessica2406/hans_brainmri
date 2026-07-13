"""
central_location.py (v5 - uses shared, tightened segmentation)

Independent (non-ResNet) predicate extractor for HAN-S.

See lesion_segmentation.py for the segmentation fix details. This
version still selects the candidate closest to image center among
survivors (v3's approach), now on top of the tightened candidate
filtering, rather than reintroducing the intensity-anomaly filter
from v4 (which debug-image inspection showed wasn't the actual
problem - the real issue was bad candidate selection upstream, not
missing anomaly detection downstream).

Usage:
    python central_location.py path/to/mri.png
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lesion_segmentation import get_lesion_candidates


def extract_central_location(
    image_path,
    threshold_value=160,
    distance_ratio_threshold=0.20,
    min_area_fraction=0.005,
    debug=True,
):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    image_center = (w / 2.0, h / 2.0)
    diagonal = np.sqrt(w ** 2 + h ** 2)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = get_lesion_candidates(gray, contours, min_area_fraction)

    if not candidates:
        return False, {
            "reason": "no lesion candidate found after segmentation filtering",
            "centroid": None,
            "normalized_distance": None,
            "debug_image": None,
        }

    scored = []
    for c in candidates:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = np.sqrt((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2)
        scored.append((dist, cx, cy, c))

    if not scored:
        return False, {
            "reason": "all candidates had degenerate moments",
            "centroid": None,
            "normalized_distance": None,
            "debug_image": None,
        }

    dist, cx, cy, lesion = min(scored, key=lambda t: t[0])
    normalized_distance = dist / diagonal

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