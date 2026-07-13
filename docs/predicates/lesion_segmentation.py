"""
lesion_segmentation.py

Shared candidate-region segmentation for all HAN-S independent predicates.

Fix vs previous per-script skull filters: those only excluded contours
that were either very large (>35% of image) or touched the image
border. Debug-image inspection (see PR discussion) showed this let a
near-whole-brain-outline contour through on at least one image
(pituitary sample) because it didn't touch the border tightly enough
and stayed just under the area cutoff, since brain outlines are
elongated/branching rather than solid, so their contour AREA (not
bounding box) can be deceptively small relative to their spatial
extent.

This version adds two additional exclusion checks:
  1. Bounding-box coverage: if a contour's bounding box covers a large
     fraction of the image (even if the contour's raw area is smaller,
     e.g. a branching/hollow outline), it's excluded as likely
     whole-brain anatomy rather than a discrete lesion.
  2. Aspect / extent sanity check: contours that are highly elongated
     or have very low "extent" (contour area relative to its own
     bounding box area) are excluded, since real lesions tend to be
     roughly compact/blob-shaped, while brain outlines and sulcal/
     ventricular patterns tend to be thin, branching, or highly
     elongated.

IMPORTANT CAVEAT: this repository's dataset provides class labels only
(no lesion location ground truth / segmentation masks), so these
heuristics are tuned by visual inspection of a handful of debug
images, not validated against ground-truth lesion locations. Real
validation requires either a dataset with expert segmentation masks
(e.g. BraTS) or Jessica's batch_validate_predicates.py results at
scale. Treat this as an improved heuristic, not a solved problem.
"""

import cv2
import numpy as np


def is_excluded_region(
    cnt,
    img_shape,
    area_fraction_limit=0.35,
    bbox_fraction_limit=0.55,
    border_margin=5,
    min_extent=0.25,
    max_aspect_ratio=6.0,
):
    """
    Returns True if a contour should be excluded as likely
    skull/whole-brain/normal-anatomy rather than a discrete lesion
    candidate.
    """
    h, w = img_shape
    img_area = h * w

    area = cv2.contourArea(cnt)
    if area <= 0:
        return True

    # 1. Raw area too large -> likely skull/whole-brain fill
    if area > area_fraction_limit * img_area:
        return True

    x, y, cw, ch = cv2.boundingRect(cnt)

    # 2. Touches image border -> likely skull/whole-brain outline
    touches_border = (
        x <= border_margin
        or y <= border_margin
        or (x + cw) >= (w - border_margin)
        or (y + ch) >= (h - border_margin)
    )
    if touches_border:
        return True

    # 3. Bounding box covers most of the image, even if raw contour
    #    area is smaller (catches branching/hollow whole-brain
    #    outlines that don't trip the area check above)
    bbox_area = cw * ch
    if bbox_area > bbox_fraction_limit * img_area:
        return True

    # 4. Low "extent" (area relative to its own bounding box) ->
    #    thin, branching, or hollow shape, not a compact lesion blob
    extent = area / bbox_area if bbox_area > 0 else 0
    if extent < min_extent:
        return True

    # 5. Extreme aspect ratio -> elongated structure (e.g. a sulcus or
    #    a strip of CSF), not a roughly blob-shaped lesion
    aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
    if aspect_ratio > max_aspect_ratio:
        return True

    return False


def get_lesion_candidates(gray, contours, min_area_fraction=0.005):
    """
    Filters a list of contours down to plausible lesion candidates
    using is_excluded_region(). Returns list of contours (not scored -
    callers apply their own scoring, e.g. largest, closest-to-center).
    """
    h, w = gray.shape
    img_area = h * w

    candidates = [
        c for c in contours
        if cv2.contourArea(c) > min_area_fraction * img_area
        and not is_excluded_region(c, (h, w))
    ]
    return candidates