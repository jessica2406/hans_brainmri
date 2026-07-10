# Independent AIL Predicates for HAN-S Brain MRI System

## Introduction

The current HAN-S (Highly Adaptive Neuro-Symbolic) framework combines a deep learning classifier with a symbolic reasoning layer for brain MRI diagnosis. The symbolic layer currently derives predicates such as `crescentic_fluid`, `well_defined_border`, and `restricted_diffusion` directly from the neural network's prediction probabilities.

Although this approach allows symbolic rules to execute, it does not provide true neuro-symbolic reasoning because the symbolic predicates are simply copies of the neural network output. As a result, the symbolic layer cannot independently validate or challenge the neural network's prediction.

This document proposes a set of independent predicate extraction techniques using classical computer vision methods. These predicates are computed directly from MRI pixel data using OpenCV-based image processing rather than from the neural network output.

---

## Problem with the Current Approach

### Existing Workflow

```
MRI Image
     │
     ▼
   ResNet
     │
     ▼
Prediction Probability
     │
     ▼
Symbolic Predicate
```

Example:

```
ResNet(Subdural) = 0.95
        ↓
crescentic_fluid = True
```

Here, the symbolic predicate is not independently computed. It merely repeats the neural network's opinion. This limits the neuro-symbolic architecture because both components always agree by construction, so the symbolic layer can never catch or flag a neural network mistake.

---

## Proposed Independent Predicate Extraction

Instead of using prediction probabilities, each predicate is extracted directly from MRI image characteristics using classical computer vision.

```
                 ┌────────► ResNet
MRI Image ───────┤
                 │
                 └────────► OpenCV
                               │
                               ▼
                     Independent Predicates
                               │
                               ▼
                    Symbolic Rule Engine
```

### 1. Crescentic Fluid Collection

**Clinical motivation:** Subdural hematomas often appear as crescent-shaped fluid collections along the inner surface of the skull.

**Proposed technique:**
- Gaussian Blur
- Intensity Thresholding
- Contour Detection
- Shape Analysis

**Expected output:** `crescentic_fluid = True / False`

### 2. Well-Defined Border

**Clinical motivation:** Meningiomas typically exhibit smooth, sharply defined borders.

**Proposed technique:**
- Canny Edge Detection
- Sobel Gradient
- Average Edge Strength Measurement

Sharp and continuous edges indicate a well-defined border.

**Expected output:** `well_defined_border = True / False`

### 3. Irregular Border

**Clinical motivation:** Gliomas often have infiltrative, irregular, and non-circular boundaries.

**Proposed technique:**
- Contour Detection
- Circularity
- Solidity
- Convex Hull Analysis

A low circularity score suggests an irregular border.

**Expected output:** `irregular_border = True / False`

### 4. Central Location (Sellar Mass)

**Clinical motivation:** Pituitary tumors usually occur near the center of the brain around the sella turcica.

**Proposed technique:**
- Segment abnormal region
- Compute contour centroid
- Measure distance from image center

If the centroid lies close to the image center, the predicate becomes true.

**Expected output:** `central_location = True / False`

---

## Example OpenCV Implementation

### Crescentic Fluid Detection

```python
gray = cv2.imread("mri.png", 0)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    blur,
    180,
    255,
    cv2.THRESH_BINARY
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

Contours can then be filtered using geometric properties such as area, eccentricity, and curvature to identify crescent-like fluid collections.

### Border Sharpness

```python
edges = cv2.Canny(gray, 50, 150)

edge_strength = np.mean(edges)

well_defined_border = edge_strength > threshold
```

Higher edge strength generally indicates a sharper lesion boundary.

### Irregular Border (Circularity)

Circularity is computed as:

```
circularity = 4 * pi * Area / Perimeter^2
```

Lower circularity indicates a more irregular border.

### Central Location (Centroid)

```python
M = cv2.moments(cnt)

cx = M["m10"] / M["m00"]
cy = M["m01"] / M["m00"]
```

`(cx, cy)` is then compared against the image center to determine `central_location`.

---

## Comparison of Approaches

| Current Method | Proposed Method |
|---|---|
| Copies ResNet probability | Extracts features directly from MRI pixels |
| Symbolic layer always agrees with neural network | Symbolic layer can independently agree or disagree |
| No independent reasoning | Independent evidence generation |
| Cannot detect neural network mistakes | Can provide supporting or conflicting evidence |

> **Note:** The table below is an illustrative placeholder showing the *type* of disagreement this approach enables. It does **not** contain real experimental results. It will be replaced with actual outputs measured on sample MRI images before this PR is finalized (see Testing section / Step 4 of the project plan).

| Image | ResNet Prediction | Independent Predicate | Status |
|---|---|---|---|
| MRI 1 | Crescentic Fluid = True | *TBD* | Placeholder — pending real test run |
| MRI 2 | Crescentic Fluid = True | *TBD* | Placeholder — pending real test run |
| MRI 3 | Crescentic Fluid = False | *TBD* | Placeholder — pending real test run |

The disagreement between the neural network and the independently extracted predicate is what enables meaningful symbolic reasoning and supports a more robust neuro-symbolic architecture.

---

## Limitations

The proposed methods rely on classical computer vision techniques and therefore have several limitations:

- MRI intensity varies across scanners and acquisition protocols.
- Threshold values may require tuning for different datasets.
- Noise may affect edge detection.
- Successful contour analysis depends on accurate segmentation.
- Texture-based methods may be sensitive to image quality.

Despite these limitations, the predicates remain independent of the neural network and therefore contribute genuine symbolic evidence.

---

## Future Work

- Adaptive thresholding techniques.
- Texture descriptors such as Gray Level Co-occurrence Matrix (GLCM).
- Active contour segmentation.
- Multi-slice or 3D MRI analysis.
- Combining several classical features into confidence scores for symbolic reasoning.

---

## Conclusion

This proposal replaces probability-derived symbolic predicates with independently extracted image-based features. Using classical computer vision methods such as thresholding, contour analysis, edge detection, and centroid estimation allows the symbolic layer to generate its own evidence directly from MRI images.

Although these techniques are less accurate than deep learning, they provide genuine independence from the neural network, enabling the HAN-S framework to perform meaningful neuro-symbolic reasoning where the symbolic layer can validate, support, or challenge neural network predictions.

---

## References

1. OpenCV Documentation – Image Processing Functions.
2. Gonzalez, R. C., & Woods, R. E. *Digital Image Processing*.
3. Szeliski, R. *Computer Vision: Algorithms and Applications*.
4. Brain MRI Radiology Literature on Meningioma, Glioma, Pituitary Adenoma, and Subdural Hematoma Imaging Characteristics.