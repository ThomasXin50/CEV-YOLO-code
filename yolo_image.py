from ultralytics import YOLO
import cv2, time
from pathlib import Path

# model = YOLO('yolo11s.pt')
model = YOLO('yolov8n-oiv7.pt')

results = model.predict(source='C:\\Users\\thoma\\Downloads\\5E5A0505.jpg', conf=0.4)[0]
annotated_frame = results.plot()
print(type(annotated_frame))
cv2.namedWindow("YOLO predicted image", cv2.WINDOW_NORMAL)
cv2.imshow("YOLO predicted image", annotated_frame)