import streamlit as st
import numpy as np
import onnxruntime as rt
from PIL import Image
import cv2
import json
import os

st.set_page_config(page_title="WheatGuard AI", layout="wide")

# Load class labels
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load ONNX model
MODEL_PATH = "model/model.onnx"
session = rt.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Preprocess function
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Prediction function
def predict(img):
    input_tensor = preprocess(img)
    output = session.run(None, {input_name: input_tensor})[0]
    probs = np.squeeze(output)
    probs = np.exp(probs) / np.sum(np.exp(probs))
    top2_idx = probs.argsort()[-2:][::-1]
    top2 = [(classes[i], probs[i]) for i in top2_idx]
    return top2

# UI
st.title("🌾 WheatGuard AI — Disease Detector")
uploaded = st.file_uploader("Upload Wheat Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    top2 = predict(img)

    st.subheader("Top Predictions:")
    for name, prob in top2:
        st.write(f"**{name} — {prob*100:.2f}%**")
        st.progress(float(prob))
