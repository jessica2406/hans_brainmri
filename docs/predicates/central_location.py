"""
central_location.py (v4 - adds anomaly filtering)

Independent (non-ResNet) predicate extractor for HAN-S.

Fix vs v3: picking the non-skull contour closest to the image center
still failed on no_tumor and pituitary images. Root cause: normal
brain anatomy (thalamus, basal ganglia, ventricles) is ALWAYS near
the geometric center of an MRI slice - that's just anatomy, not
pathology. So "closest blob to center" finds *something* central on
every scan, including scans with no lesion at all.

This version adds an intensity-anomaly filter: a real lesion usually
has a mean intensity that differs noticeably from the surrounding
background brain tissue (edema, necrosis, or a mass are typically
brighter or darker than normal parenchyma), whereas normal central
structures blend in with the rest of the brain on a T1/T2 slice.
Candidates are now required to pass BOTH a centrality check and an
intensity-contrast check against a background sample ring around
each candidate.

NOTE: this is still a coarse heuristic calibrated on a handful of
images per class - not a substitute for validation against a larger
sample. Flagged explicitly as such in the report.

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


def _intensity_contrast(gray, cnt):
    """
    Returns |mean intensity inside contour - mean intensity in a
    surrounding ring| as a fraction of overall image intensity range.
    Higher = more anomalous relative to local background.
    """
    mask_inner = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_inner, [cnt], -1, 255, thickness=-1)

    dilated = cv2.dilate(mask_inner, np.ones((25, 25), np.uint8))
    ring = cv2.subtract(dilated, mask_inner)

    if np.count_nonzero(mask_inner) == 0 or np.count_nonzero(ring) == 0:
        return 0.0

    inner_mean = float(np.mean(gray[mask_inner == 255]))
    ring_mean = float(np.mean(gray[ring == 255]))

    value_range = max(float(np.max(gray)) - float(np.min(gray)), 1.0)
    return abs(inner_mean - ring_mean) / value_range


def extract_central_location(
    image_path,
    threshold_value=160,
    distance_ratio_threshold=0.20,
    min_area_fraction=0.005,
    contrast_threshold=0.12,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Threshold + contour detection, excluding skull contours.
      2. For each remaining candidate, compute intensity contrast
         against its local surrounding background.
      3. Discard candidates below contrast_threshold - these are
         treated as normal anatomy, not a distinct lesion.
      4. Among surviving (anomalous) candidates, pick the one closest
         to the image center.
      5. central_location = True if that candidate's normalized
         centroid distance is below distance_ratio_threshold.
      6. If no candidate survives both filters, return False.
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

    base_candidates = [
        c for c in contours
        if cv2.contourArea(c) > min_area_fraction * img_area
        and not _is_probably_skull(c, (h, w))
    ]

    anomalous_candidates = []
    for c in base_candidates:
        contrast = _intensity_contrast(gray, c)
        if contrast >= contrast_threshold:
            anomalous_candidates.append((c, contrast))

    if not anomalous_candidates:
        return False, {
            "reason": "no intensity-anomalous (non-normal-anatomy) candidate found",
            "num_base_candidates": len(base_candidates),
            "num_anomalous_candidates": 0,
            "centroid": None,
            "normalized_distance": None,
            "debug_image": None,
        }

    scored = []
    for c, contrast in anomalous_candidates:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = np.sqrt((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2)
        scored.append((dist, cx, cy, c, contrast))

    if not scored:
        return False, {
            "reason": "all anomalous candidates had degenerate moments",
            "centroid": None,
            "normalized_distance": None,
            "debug_image": None,
        }

    dist, cx, cy, lesion, contrast = min(scored, key=lambda t: t[0])
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
        "intensity_contrast": round(contrast, 3),
        "num_base_candidates": len(base_candidates),
        "num_anomalous_candidates": len(anomalous_candidates),
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
    print(f"  intensity_contrast: {details.get('intensity_contrast')}")
    print(f"  base candidates: {details.get('num_base_candidates')}, anomalous: {details.get('num_anomalous_candidates')}")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")