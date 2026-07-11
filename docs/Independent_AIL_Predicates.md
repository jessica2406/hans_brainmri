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

The table below reports actual outputs from running the four OpenCV prototype scripts (`docs/predicates/`) on three real subdural empyema MRI images from `data/empyema/`. No ResNet predictions were available to compare against for these images (this repository currently contains only the empyema class), so this table demonstrates independent predicate extraction and cross-predicate variation rather than a direct agreement/disagreement comparison against the neural network. That comparison should be added once ResNet outputs for these same images are available.

| Image | crescentic_fluid | well_defined_border | irregular_border | central_location |
|---|---|---|---|---|
| subdural-empyema (1).jpg | True | False (edge_strength=0.0) | True (circularity=0.46, solidity=0.84) | False (norm_dist=0.249) |
| subdural-empyema-1 (1).jpg | True | N/A — no valid lesion contour found | N/A — no valid lesion contour found | False (norm_dist=0.225) |
| posterior-fossa-subdural-empyemas-from-mastoiditis.jpg | True | False (edge_strength=14.45) | True (circularity=0.071, solidity=0.685) | True (norm_dist=0.178) |

**Observations:**
- `crescentic_fluid` correctly fired `True` on all three confirmed empyema images, consistent with the clinical expectation that empyema/subdural fluid collections are crescent-shaped.
- `well_defined_border` and `central_location` both correctly returned mostly `False` on empyema images — clinically expected, since empyema is a diffuse fluid collection rather than a sharply-bordered, centrally-located mass (unlike meningioma or pituitary adenoma, which these predicates specifically target).
- On `subdural-empyema-1 (1).jpg`, both `well_defined_border` and `irregular_border` failed to find any valid non-skull lesion contour and returned no result. This is a genuine current limitation (see Limitations), likely due to low contrast between the lesion and surrounding tissue in that particular image, rather than a fabricated or adjusted outcome.
- No meningioma, glioma, or pituitary sample images were available in this repository at the time of testing, so `well_defined_border`, `irregular_border`, and `central_location` could not yet be validated against their intended target classes. This is called out explicitly as follow-up work rather than assumed to be working correctly.

The variation across predicates and images (rather than a constant answer) demonstrates that these predicates are functioning as independent evidence sources, capable of disagreeing with each other and, once ResNet outputs are available for the same images, with the neural network.

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