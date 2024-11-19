# signdetection
1. Introduction
This report details the process of training, validating, and evaluating a YOLOv8 model for Sign Language Detection using a custom dataset. The goal was to detect six sign classes (iloveyou, yes, no, thankyou, ok, hello) with robust accuracy and explainability. The report includes metrics, visual outputs, and iterative improvements.
________________________________________
2. Dataset Overview
Dataset Summary
•	Image Resolution: 320x320 pixels.
•	Number of Classes: 6.
•	Training Set: 112 images.
•	Validation Set: 9 images.
•	Test Set: 10 images.
Class Labels:
•	iloveyou, yes, no, thankyou, ok, hello.
Data Augmentation
To improve model performance, the following augmentations were applied during training:
•	Horizontal flipping.
•	Rotation (random within ±30 degrees).
________________________________________
3. Training Process
Model Configuration
•	Model: YOLOv8 Nano (yolov8n.pt).
•	Image Size: 320x320.
•	Batch Size: 8.
•	Epochs: 30.
•	Learning Rate: 0.001 (with cosine decay scheduler).
Training Metrics
During training, the following metrics were tracked:
•	Box Loss: Measures localization accuracy of bounding boxes.
•	Classification Loss: Accuracy of class predictions.
•	DFL Loss: Distribution Focal Loss for objectness.
Training Command
python
Copy code
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
Training Performance
Below are the training results visualized over 30 epochs.
Key Observations:
•	Train/Box Loss: Steady decline, showing improved bounding box predictions.
•	Train/Cls Loss: Significant reduction over epochs, indicating better class prediction.
•	Val/mAP@0.5: Peaked at 90% by the 25th epoch.
•	Val/mAP@0.5:0.95: Gradual improvement, reaching ~78% after 30 epochs.
________________________________________
4. Validation and Testing
Confusion Matrix
The confusion matrix evaluates class-level accuracy for predictions on the test set.
1.	Confusion Matrix (Raw Counts):
 Observations:
o	True positives for ok dominate, with minor misclassifications in thankyou and hello.
2.	Normalized Confusion Matrix:
 Observations:
o	Precision and recall for hello and ok are highest, whereas thankyou had moderate confusion with other classes.
F1-Confidence Curve
 Key Observations:
•	Optimal Confidence: 0.92 achieved for all classes combined.
•	Per-Class Peaks:
o	ok: Highest F1 score due to minimal misclassifications.
o	no: Lower F1 score due to fewer samples.
________________________________________
5. Improvements
1.	Addressed Overfitting:
o	Applied data augmentations (rotation, flipping) to diversify training samples.
2.	Hyperparameter Optimization:
o	Adjusted learning rate to prevent stagnation in training.
3.	Iterative Validation:
o	Monitored metrics at each epoch to refine stopping criteria.
________________________________________
6. Conclusions
•	Best Performance:
o	mAP@0.5: 90%.
o	mAP@0.5:0.95: 78%.
•	Challenges:
o	Limited dataset size caused occasional class confusion.
o	Class imbalance slightly reduced performance for underrepresented classes.
•	Future Steps:
o	Collect more diverse samples to improve generalization.
o	Experiment with larger models like yolov8s.pt for improved accuracy.
