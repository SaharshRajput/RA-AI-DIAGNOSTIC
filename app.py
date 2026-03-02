import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="RA AI Diagnostic", layout="wide")
st.title("🦴 RA Severity Predictor & Joint Analysis")

# --- MODEL LOADING ---
device = torch.device("cpu") 
labels = ['Severe', 'Moderate', 'Healthy']

@st.cache_resource
def load_model():
    model_path = 'model.pth' 
    
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure 'model.pth' is uploaded to GitHub.")
        st.stop()
            
    # Case 2: Use the Fine-Tuning architecture
    model = models.resnet18(weights=None) # We use None because we are loading YOUR weights
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    
    # Load your trained weights
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

# --- IMAGE PROCESSING (CASE 1: CLAHE) ---
def apply_clahe(img):
    img_np = np.array(img)
    # Convert to Grayscale for CLAHE
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    
    # Convert back to RGB for ResNet
    final_img = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_img)

# --- GRAD-CAM LOGIC ---
def get_analysis(img_file, model):
    # 1. Load and Enhance
    img_original = Image.open(img_file).convert('RGB')
    img_enhanced = apply_clahe(img_original)
    
    target_layer = model.layer4[-1]
    features = []
    def hook_feat(module, input, output): features.append(output)
    handle = target_layer.register_forward_hook(hook_feat)
    
    # 2. Preprocess
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = t(img_enhanced).unsqueeze(0).to(device)
    
    # 3. Prediction
    output = model(input_tensor)
    handle.remove()
    
    # 4. Create Heatmap
    weights = torch.mean(features[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features[0], dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cv2.resize(cam_img, (224, 224)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap, output, img_enhanced

# --- UI INTERACTION ---
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner('AI analyzing joint structures...'):
        model = load_model()
        heatmap, output, enhanced_img = get_analysis(uploaded_file, model)
        
        prob = torch.nn.functional.softmax(output, dim=1)
        score, idx = torch.max(prob, 1)
        res_label = labels[idx.item()]
        res_conf = score.item() * 100

    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Enhanced X-ray")
        st.image(enhanced_img, use_container_width=True, caption="CLAHE Applied")
        st.metric(label="PREDICTED SEVERITY", value=res_label)

    with col2:
        st.subheader("2. AI Focus (Heatmap)")
        st.image(heatmap, use_container_width=True, caption="Analysis Hotspots")
        
    with col3:
        st.subheader("3. Diagnostic Confidence")
        # Horizontal Bar Chart for Case 1 & 2 breakdown
        for i, label in enumerate(labels):
            confidence = prob[0][i].item()
            st.write(f"**{label}** ({confidence*100:.1f}%)")
            st.progress(confidence)
            
        # Low Confidence Warning
        top_two = torch.topk(prob, 2)
        diff = top_two.values[0][0] - top_two.values[0][1]
