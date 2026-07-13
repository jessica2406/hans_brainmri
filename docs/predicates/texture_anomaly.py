"""
texture_anomaly.py (v2 - requires spatial clustering)

Fix vs v1: v1 flagged True on every test image because normal brain
texture (gray/white matter boundaries, cortical folding, ventricle
edges) naturally varies enough that SOME individual window will
random-cross a low z-score threshold on almost any scan. Debug
inspection showed dozens of scattered "anomalous" windows across
normal tissue, not a signal localized to any one region.

Real pathology should show up as a spatially CONTIGUOUS cluster of
texturally anomalous windows (a mass has a size and shape), not
isolated one-off windows scattered around the brain. This version:
  1. Raises the z-score threshold substantially (was 2.0, now 3.0).
  2. Requires a minimum cluster of adjacent anomalous windows
     (connected-component analysis on the anomaly mask) before firing
     True, rather than accepting any single anomalous window.

Usage:
    python texture_anomaly.py path/to/mri.png
"""

import sys
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage


def _brain_mask(gray, threshold_value=20):
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return mask


def _glcm_features(patch, levels=32):
    patch_scaled = (patch.astype(np.float64) / 255.0 * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(
        patch_scaled,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    return np.array([contrast, homogeneity, energy, correlation])


def extract_texture_anomaly(
    image_path,
    window_size=32,
    stride=16,
    zscore_threshold=3.0,
    min_cluster_windows=4,
    min_brain_fraction=0.6,
    debug=True,
):
    """
    Returns (predicate: bool, details: dict)

    Method:
      1. Slide a window across brain tissue, compute GLCM texture
         features per window.
      2. z-score each window's features against the image's own
         average.
      3. Mark windows exceeding zscore_threshold as "anomalous".
      4. Run connected-component analysis on the anomalous-window grid
         to find spatially contiguous clusters.
      5. texture_anomaly = True only if the LARGEST cluster contains
         at least min_cluster_windows adjacent anomalous windows -
         isolated single anomalous windows (likely normal texture
         variation) are ignored.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    mask = _brain_mask(gray)

    grid_positions = []  # (grid_row, grid_col, x, y)
    features = []

    y_steps = list(range(0, h - window_size, stride))
    x_steps = list(range(0, w - window_size, stride))

    for row, y in enumerate(y_steps):
        for col, x in enumerate(x_steps):
            patch = gray[y:y + window_size, x:x + window_size]
            mask_patch = mask[y:y + window_size, x:x + window_size]
            brain_fraction = np.count_nonzero(mask_patch) / mask_patch.size
            if brain_fraction < min_brain_fraction:
                continue
            feats = _glcm_features(patch)
            grid_positions.append((row, col, x, y))
            features.append(feats)

    if len(features) < 10:
        return False, {
            "reason": "too few valid brain-tissue windows to compute a baseline",
            "num_windows": len(features),
            "debug_image": None,
        }

    features = np.array(features)
    mean_vec = features.mean(axis=0)
    std_vec = features.std(axis=0)
    std_vec[std_vec == 0] = 1e-6

    z_scores = np.abs((features - mean_vec) / std_vec)
    max_z_per_window = z_scores.max(axis=1)
    is_anomalous = max_z_per_window > zscore_threshold

    # Build a grid to run connected-component clustering on
    n_rows = len(y_steps)
    n_cols = len(x_steps)
    anomaly_grid = np.zeros((n_rows, n_cols), dtype=bool)
    for (row, col, x, y), anomalous in zip(grid_positions, is_anomalous):
        if anomalous:
            anomaly_grid[row, col] = True

    labeled, num_clusters = ndimage.label(anomaly_grid, structure=np.ones((3, 3)))
    cluster_sizes = ndimage.sum(anomaly_grid, labeled, range(1, num_clusters + 1)) if num_clusters > 0 else []
    largest_cluster_size = int(max(cluster_sizes)) if len(cluster_sizes) > 0 else 0

    predicate = largest_cluster_size >= min_cluster_windows

    # Find the windows belonging to the largest cluster, for debug drawing
    largest_cluster_windows = []
    if largest_cluster_size > 0:
        largest_label = int(np.argmax(cluster_sizes)) + 1
        for (row, col, x, y) in grid_positions:
            if labeled[row, col] == largest_label:
                largest_cluster_windows.append((x, y))

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (row, col, x, y), anomalous in zip(grid_positions, is_anomalous):
            color = (0, 0, 255) if anomalous else (60, 60, 60)
            cv2.rectangle(debug_img, (x, y), (x + window_size, y + window_size), color, 1)
        for (x, y) in largest_cluster_windows:
            cv2.rectangle(debug_img, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
        base, ext = os.path.splitext(image_path)
        debug_path = f"{base}_texture_debug.png"
        cv2.imwrite(debug_path, debug_img)
    else:
        debug_path = None

    details = {
        "num_windows": len(features),
        "num_anomalous_windows": int(np.sum(is_anomalous)),
        "num_clusters": int(num_clusters),
        "largest_cluster_size": largest_cluster_size,
        "min_cluster_windows_required": min_cluster_windows,
        "debug_image": debug_path,
    }

    return predicate, details


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python texture_anomaly.py path/to/mri.png")
        sys.exit(1)

    image_path = sys.argv[1]
    predicate, details = extract_texture_anomaly(image_path)

    print(f"texture_anomaly = {predicate}")
    print(f"  windows analyzed: {details.get('num_windows')}")
    print(f"  anomalous windows: {details.get('num_anomalous_windows')}")
    print(f"  largest cluster size: {details.get('largest_cluster_size')} (need >= {details.get('min_cluster_windows_required')})")
    if details.get("debug_image"):
        print(f"  debug image saved: {details['debug_image']}")