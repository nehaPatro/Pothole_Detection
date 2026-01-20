import gradio as gr
import numpy as np
import cv2
from PIL import Image
import torch

# YOLOv12 imports
from yolov12.models.common import DetectMultiBackend
from yolov12.utils.general import non_max_suppression, scale_boxes
from yolov12.utils.torch_utils import select_device

# Load model
def load_model():
    device = select_device("cpu")

    model = DetectMultiBackend(
        "best.pt",
        device=device,
        dnn=False,
        fp16=False
    )
    model.eval()
    return model, device


model, device = load_model()


def predict(image):
    img0 = np.array(image)

    # preprocess
    img = cv2.resize(img0, (640, 640))
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # draw boxes
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
                    0.6,
                    (0, 255, 0),
                    2,
                )

    return Image.fromarray(img0)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Pothole Detection using YOLOv12",
    description="Upload a road image to detect potholes",
)

demo.launch()
