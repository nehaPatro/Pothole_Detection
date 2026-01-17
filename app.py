import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch

# YOLOv12 imports
from yolov12.models.common import DetectMultiBackend
from yolov12.utils.general import non_max_suppression, scale_boxes
from yolov12.utils.torch_utils import select_device

st.set_page_config(page_title="Pothole Detection", layout="centered")
st.title("🕳️ Pothole Detection using YOLOv12")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    device = select_device("cpu")  # Streamlit Cloud = CPU
    model = DetectMultiBackend(
        "weights/best.pt",
        device=device,
        dnn=False,
        fp16=False
    )
    model.eval()
    return model, device

model, device = load_model()

# ------------------ Upload Image ------------------
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img0 = np.array(image)

    # Preprocess
    img = cv2.resize(img0, (640, 640))
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Draw boxes
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img0,
                    f"Pothole {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    st.image(img0, caption="Detected Potholes", use_column_width=True)
