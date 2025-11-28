import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import json

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="WheatGuard AI",
    page_icon="🌾",
    layout="wide",
)

# Sidebar Branding
with st.sidebar:
    st.title("🌾 WheatGuard AI")
    st.markdown("Made with ❤️ by **Nikhil & Sanket**")

    st.markdown("### Navigate")
    st.page_link("app.py", label="Detector", icon="🧪")
    st.markdown("---")
    st.info("Upload wheat leaf images to detect diseases.")

# ------------------------------
# Load Class Map
# ------------------------------
CLASS_MAP_PATH = "class_map.json"

with open(CLASS_MAP_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Convert keys to int if needed
CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}

NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = "model/model.pth"

@st.cache_resource
def load_model():
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ------------------------------
# Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def predict(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]
    return probs


# ------------------------------
# UI Layout
# ------------------------------
st.title("🌾 WheatGuard AI — Smart Leaf Disease Detection")
st.markdown("Upload an image of a wheat leaf to detect disease using **AI (DeiT Transformer)**")

uploaded_files = st.file_uploader(
    "Upload one or multiple images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(2)
    idx = 0

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        probs = predict(img)

        top3 = torch.topk(probs, 3)
        pred_idx = top3.indices.tolist()
        pred_scores = top3.values.tolist()

        with st.container(border=True):
            if idx % 2 == 0:
                with cols[0]:
                    st.image(img, caption=f"Uploaded Image", use_column_width=True)
            else:
                with cols[1]:
                    st.image(img, caption=f"Uploaded Image", use_column_width=True)

            st.subheader("Prediction Results")

            for i in range(3):
                disease = CLASS_NAMES[pred_idx[i]]
                confidence = pred_scores[i] * 100

                st.write(f"**{i+1}. {disease} — {confidence:.2f}%**")
                st.progress(float(confidence / 100))

            idx += 1
