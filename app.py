import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request
# --- PAGE SETUP ---
st.set_page_config(page_title="RA AI Diagnostic", layout="wide")
st.title("🦴 RA Severity Predictor & Joint Analysis")

# --- MODEL LOADING ---
device = torch.device("cpu") # Use CPU for stable web deployment
labels = ['Healthy', 'Moderate', 'Severe']



@st.cache_resource
def load_model():
    model_path = 'model.pth' # The file is now local!
          # In Colab, change your model definition to this:
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # This freezes the early layers so the AI keeps its "knowledge" of shapes
    # but learns the "specifics" of your X-rays in the last layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer for your 3 classes (Healthy, Moderate, Severe)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    
    # When you train, only the last layer will learn initially
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure 'model.pth' is uploaded to GitHub.")
        st.stop()
            
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    
    # Crucial: weights_only=False for PyTorch 2.6 compatibility
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(state_dict)
    
    model.eval()
    return model
# --- GRAD-CAM LOGIC (THE ANALYSIS) ---
def get_analysis(img_file, model):
    def apply_clahe(img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    
    # Convert back to RGB for ResNet
    final_img = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_img)
        
    # Convert PIL to Open CV format
    
    # Convert to Grayscale if it's RGB
    i

# --- Update your transform block ---
img = Image.open(img_file).convert('RGB')
img = apply_clahe(img) # Add this line before transforms
    target_layer = model.layer4[-1]
    features = []
    def hook_feat(module, input, output): features.append(output)
    handle = target_layer.register_forward_hook(hook_feat)
    
    # Preprocess
    img = Image.open(img_file).convert('RGB')
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = t(img).unsqueeze(0).to(device)
    
    # Run Prediction
    output = model(input_tensor)
    handle.remove()
    
    # Create Heatmap
    weights = torch.mean(features[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features[0], dim=1).squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cv2.resize(cam_img, (224, 224)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap, output

# --- UI INTERACTION ---
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Show the user we are working
    with st.spinner('AI analyzing joint structures...'):
        model = load_model()
        heatmap, output = get_analysis(uploaded_file, model)
        
        # Calculate scores
        prob = torch.nn.functional.softmax(output, dim=1)
        score, idx = torch.max(prob, 1)
        res_label = labels[idx.item()]
        res_conf = score.item() * 100

    # 2. FORCE PRINT TO SCREEN
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, width=450)
        # Professional big-text result
        st.metric(label="RA SEVERITY", value=res_label)
        st.write(f"**Certainty:** {res_conf:.1f}%")

    with col2:
        st.subheader("AI Focus (Heatmap)")
        st.image(heatmap, width=450, caption="Red areas show where AI detected symptoms")
        st.info("The AI specifically analyzes the joint spaces for narrowing and bone erosion.")

    st.success("✅ Analysis successfully rendered.")
    # Create a horizontal bar chart for probabilities
st.subheader("Diagnostic Confidence")
for i, label in enumerate(labels):
    score = prob[0][i].item()
    st.write(f"**{label}**")
    st.progress(score) # This creates a visual bar
    st.write(f"{score*100:.1f}%")
