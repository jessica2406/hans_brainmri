import streamlit as st
import torch
import numpy as np
import cv2
import os
from model import get_model
from medical_logic import SymbolicReasoner
from visualizer import get_gradcam, overlay_heatmap

st.set_page_config(page_title="HAN-S Neuro-Symbolic AI", layout="wide")
st.title("🧠 HAN-S: Subdural Empyema & Tumor Diagnostic System")

@st.cache_resource
def load_model():
    model = get_model(num_classes=5)
    if os.path.exists("brain_model_neural.pth"):
        model.load_state_dict(torch.load("brain_model_neural.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
reasoner = SymbolicReasoner()

uploaded_file = st.file_uploader("Upload Brain MRI (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Process Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_norm = cv2.resize(img_raw, (224, 224)) / 255.0
    img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0)

    # 2. Neural Prediction
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_class = torch.argmax(probs).item()
    
    # 3. Generate Heatmap
    heatmap = get_gradcam(model, img_tensor, pred_class)
    visual_proof = overlay_heatmap(img_norm, heatmap)

    # 4. Symbolic Reasoning (AIL Layer)
    class_names = ['empyema', 'no_tumor', 'glioma', 'meningioma', 'pituitary']
    detected_features = {
        "crescentic_fluid": probs[0].item(), "restricted_diffusion": probs[0].item(),
        "well_defined_border": probs[3].item(), "solid_mass": probs[3].item(),
        "irregular_border": probs[2].item(), "central_location": probs[4].item(),
        "brain_symmetry": probs[1].item()
    }
    diagnosis, proof = reasoner.reason(detected_features)

    # 5. UI Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Scan")
        st.image(img_raw, use_container_width=True)
    with col2:
        st.subheader("Visual Proof (Heatmap)")
        st.image(visual_proof, use_container_width=True)
        st.caption("Red areas indicate where the AI focused its diagnosis.")
    with col3:
        st.subheader("Logical Analysis")
        st.metric("Top Prediction", diagnosis.upper(), f"{probs[pred_class]:.2%}")
        for step in proof:
            st.write(step)