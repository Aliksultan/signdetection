# signdetection
# YOLO Model Training and Evaluation Report

---

## 1. Introduction
This report details the process of training, validating, and evaluating a YOLOv8 model for **Sign Language Detection**. The objective was to detect six sign classes (`iloveyou`, `yes`, `no`, `thankyou`, `ok`, `hello`) with robust accuracy. The report includes metrics, visual outputs, and iterative improvements.

---

## 2. Dataset Overview

### Dataset Summary
- **Image Resolution**: `320x320` pixels.
- **Number of Classes**: 6.
- **Training Set**: 112 images.
- **Validation Set**: 9 images.
- **Test Set**: 10 images.

### Class Labels:
`iloveyou`, `yes`, `no`, `thankyou`, `ok`, `hello`.

### Data Augmentation
To improve model performance, the following augmentations were applied:
- Horizontal flipping.
- Rotation (±30 degrees).

---

## 3. Training Process

### Model Configuration
- **Model**: YOLOv8 Nano (`yolov8n.pt`).
- **Image Size**: `320x320`.
- **Batch Size**: 8.
- **Epochs**: 30.
- **Learning Rate**: 0.001 (with cosine decay scheduler).

### Training Metrics
The following metrics were tracked during training:
- **Box Loss**: Localization accuracy of bounding boxes.
- **Classification Loss**: Accuracy of class predictions.
- **DFL Loss**: Distribution Focal Loss for objectness.

### Training Command
```python
from ultralytics import YOLO

# Load and train the model
model = YOLO("yolov8n.pt")
model.train(
    data="dataset.yaml",
    epochs=30,
    imgsz=320,
    batch=8,
    name="sign_detection_model",
    val=True
)

