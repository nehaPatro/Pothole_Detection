import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("Pothole Detection System")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload section
uploaded_file = st.file_uploader(
    "Upload road image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Prediction
    results = model(img)

    for r in results:
        output = r.plot()

    st.image(output, caption="Detected Potholes")
