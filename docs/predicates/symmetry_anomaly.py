"""
symmetry_anomaly.py (v2 - erodes mask to exclude boundary artifacts)

Fix vs v1: debug image inspection showed the flagged "anomalous"
cluster on the glioma test image was sitting right at the skull/scalp
edge, not in actual brain tissue - likely because the brain mask
includes a thin rim of skull/boundary pixels, and comparing that rim
to its mirror produces spurious high-difference scores unrelated to
pathology (skull thickness, scan edge artifacts, and minor head-tilt
misalignment all show up here).

Fix: erode the brain mask inward by a margin before using it to
select valid block pairs, so blocks are only accepted well inside
actual brain tissue, away from the skull boundary.

Usage:
    python symmetry_anomaly.py path/to/mri.png
"""

import sys
import os
import cv2
import numpy as np
from scipy import ndimage


def _brain_mask(gray, threshold_value=20, erosion_margin=15):
    """Threshold + erode inward to exclude skull/boundary rim pixels."""
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((erosion_margin, erosion_margin), np.uint8)
    eroded = cv2.erode(mask, kernel)
    return eroded


def _find_midline_x(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask.shape[1] // 2
    return int(np.mean(xs))


def extract_symmetry_anomaly(
    image_path,
    block_size=24,
    zscore_threshold=2.5,
    min_cluster_blocks=4,
    min_brain_fraction=0.8,
    erosion_margin=15,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Same method as v1, but:
      - brain mask is eroded inward by erosion_margin pixels before use,
        excluding skull/scalp boundary rim from consideration
      - min_brain_fraction raised (0.5 -> 0.8) since we now require
        blocks to be solidly inside eroded brain tissue, not partially
        at an edge
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    mask = _brain_mask(gray, erosion_margin=erosion_margin)
    midline_x = _find_midline_x(mask)

    half_width = min(midline_x, w - midline_x)
    if half_width < block_size * 2:
        return False, {
            "reason": "brain region too narrow relative to midline for symmetry analysis",
            "debug_image": None,
        }

    grid_positions = []
    diff_scores = []

    y_steps = list(range(0, h - block_size, block_size))
    x_steps = list(range(0, half_width - block_size, block_size))

    for row, y in enumerate(y_steps):
        for col, x_offset in enumerate(x_steps):
            lx = midline_x - x_offset - block_size
            rx = midline_x + x_offset

            if lx < 0 or (rx + block_size) > w:
                continue

            left_block = gray[y:y + block_size, lx:lx + block_size]
            right_block = gray[y:y + block_size, rx:rx + block_size]
            right_block_flipped = np.fliplr(right_block)

            left_mask = mask[y:y + block_size, lx:lx + block_size]
            right_mask = mask[y:y + block_size, rx:rx + block_size]
            brain_frac = min(
                np.count_nonzero(left_mask) / left_mask.size,
                np.count_nonzero(right_mask) / right_mask.size,
            )
            if brain_frac < min_brain_fraction:
                continue

            diff = float(np.mean(np.abs(
                left_block.astype(np.float64) - right_block_flipped.astype(np.float64)
            )))

            grid_positions.append((row, col, lx, y))
            diff_scores.append(diff)

    if len(diff_scores) < 10:
        return False, {
            "reason": "too few valid symmetric block pairs to compute a baseline",
            "num_blocks": len(diff_scores),
            "debug_image": None,
        }

    diff_scores = np.array(diff_scores)
    mean_diff = diff_scores.mean()
    std_diff = diff_scores.std()
    std_diff = std_diff if std_diff > 1e-6 else 1e-6

    z_scores = (diff_scores - mean_diff) / std_diff
    is_anomalous = z_scores > zscore_threshold

    n_rows = len(y_steps)
    n_cols = len(x_steps)
    anomaly_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for (row, col, lx, y), anomalous in zip(grid_positions, is_anomalous):
        if anomalous:
            anomaly_grid[row, col] = True

    labeled, num_clusters = ndimage.label(anomaly_grid, structure=np.ones((3, 3)))
    cluster_sizes = ndimage.sum(anomaly_grid, labeled, range(1, num_clusters + 1)) if num_clusters > 0 else []
    largest_cluster_size = int(max(cluster_sizes)) if len(cluster_sizes) > 0 else 0

    predicate = largest_cluster_size >= min_cluster_blocks

    largest_cluster_blocks = []
    if largest_cluster_size > 0:
        largest_label = int(np.argmax(cluster_sizes)) + 1
        for (row, col, lx, y) in grid_positions:
            if labeled[row, col] == largest_label:
                largest_cluster_blocks.append((lx, y))

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(debug_img, (midline_x, 0), (midline_x, h), (255, 255, 0), 1)
        for (row, col, lx, y), anomalous in zip(grid_positions, is_anomalous):
            rx = midline_x + (col * block_size)
            color = (0, 0, 255) if anomalous else (60, 60, 60)
            cv2.rectangle(debug_img, (lx, y), (lx + block_size, y + block_size), color, 1)
            cv2.rectangle(debug_img, (rx, y), (rx + block_size, y + block_size), color, 1)
        for (lx, y) in largest_cluster_blocks:
            cv2.rectangle(debug_img, (lx, y), (lx + block_size, y + block_size), (0, 255, 0), 2)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_symmetry_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "midline_x": midline_x,
        "num_block_pairs": len(diff_scores),
        "num_anomalous_blocks": int(np.sum(is_anomalous)),
        "largest_cluster_size": largest_cluster_size,
        "min_cluster_blocks_required": min_cluster_blocks,
        "debug_image": debug_path,
    }

    return predicate, details


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python symmetry_anomaly.py path/to/mri.png")
        sys.exit(1)

    image_path = sys.argv[1]
    predicate, details = extract_symmetry_anomaly(image_path)

    print(f"symmetry_anomaly = {predicate}")
    print(f"  midline_x: {details.get('midline_x')}")
    print(f"  block pairs analyzed: {details.get('num_block_pairs')}")
    print(f"  anomalous blocks: {details.get('num_anomalous_blocks')}")
    print(f"  largest cluster size: {details.get('largest_cluster_size')} (need >= {details.get('min_cluster_blocks_required')})")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")