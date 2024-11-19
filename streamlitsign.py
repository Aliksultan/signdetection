import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.title("YOLOv8 Live Detection with Streamlit")

st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("YOLO Model Path", r"C:\Users\aliko\Downloads\final\runs\detect\yolov8-custom-model4\weights\best.pt")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
camera_index = st.sidebar.number_input("Camera Index", value=0, step=1)

# Load YOLO Model
if os.path.exists(model_path):
    model = YOLO(model_path)
    st.sidebar.success("Model Loaded Successfully")
else:
    st.sidebar.error("Model path is invalid. Please provide a valid path.")

# Start Detection Button
start_detection = st.sidebar.button("Start Detection")

# Temporary File for Video Frames
temp_dir = tempfile.TemporaryDirectory()
temp_video_path = os.path.join(temp_dir.name, "output.mp4")

if start_detection:
    stframe = st.empty()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        st.stop()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frame detected. Exiting...")
            break

        results = model.predict(frame, imgsz=320, conf=confidence_threshold)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{model.names[cls]} {conf*100:.2f}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)


    cap.release()
    temp_dir.cleanup()
