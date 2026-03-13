import torch
import numpy as np
from model import get_model
from medical_logic import SymbolicReasoner
import cv2
import os

def run_hans_inference(path):
    # 1. Load the "Neural" Brain
    device = torch.device("cpu")
    model = get_model(num_classes=5)
    model.load_state_dict(torch.load("brain_model_neural.pth", map_location=device))
    model.to(device)
    model.eval()

    # 2. Universal Loader (Handles .npy or .jpg/.png)
    if path.endswith('.npy'):
        img_resized = np.load(path)
        # .npy files are already normalized and resized
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read file at {path}")
            return
        img_resized = cv2.resize(img, (224, 224)) / 255.0

    # Convert to Tensor [1, 1, 224, 224]
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)

    # 3. Neural Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]  # Probabilities for all 5 classes

    # Class Order:
    # 0: empyema
    # 1: no_tumor
    # 2: glioma
    # 3: meningioma
    # 4: pituitary

    # 4. Map Neural Outputs to Symbolic Predicates
    detected_features = {
        # Empyema
        "crescentic_fluid": probs[0].item(),
        "restricted_diffusion": probs[0].item() * 0.9,
        "peripheral_enhancement": probs[0].item() * 1.1,

        # Meningioma
        "well_defined_border": probs[3].item(),
        "dural_tail": probs[3].item() * 0.95,
        "solid_mass": probs[3].item() * 1.05,

        # Glioma
        "irregular_border": probs[2].item(),
        "infiltrative_growth": probs[2].item() * 1.1,

        # Pituitary
        "central_location": probs[4].item(),
        "sellar_mass": probs[4].item() * 1.2,

        # No Tumor
        "brain_symmetry": probs[1].item(),
        "clear_sulci": probs[1].item() * 0.9
    }

    # 5. Symbolic Reasoning
    reasoner = SymbolicReasoner()
    diagnosis, proof = reasoner.reason(detected_features)

    # 6. Print Results
    print("-" * 40)
    print(f"FILE: {os.path.basename(path)}")
    print(f"AI DIAGNOSIS: {diagnosis.upper()}")
    print("\nCLASS PROBABILITIES:")
    print(f" Empyema     : {probs[0].item():.2%}")
    print(f" No Tumor    : {probs[1].item():.2%}")
    print(f" Glioma      : {probs[2].item():.2%}")
    print(f" Meningioma  : {probs[3].item():.2%}")
    print(f" Pituitary   : {probs[4].item():.2%}")

    print("\nLOGICAL PROOF TREE:")
    for step in proof:
        print(f" - {step}")
    print("-" * 40)


if __name__ == "__main__":
    test_image = "data/processed/empyema/emp_aug_1_4.npy"
    try:
        run_hans_inference(test_image)
    except Exception as e:
        print(f"Error: {e}. Make sure to put a valid image path!")
