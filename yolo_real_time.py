from ultralytics import YOLO
import cv2, time
from pathlib import Path

# model = YOLO('yolo11s.pt')
model = YOLO('yolov8n-oiv7.pt') # yolo v8 nano pretrained on Google v7 dataset
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open video stream")
    exit()

frame_count = 0
start_time = time.time()

print("Starting real-time object detection. Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()

    frame_count +=1
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000/frame_count
    fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0

    perf_text = f"FPS: {fps:.2f} | Time/frame: {inference_time_ms:.2f} ms"
    cv2.putText(annotated_frame, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Real-Time Performance", annotated_frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if frame_count >= 100:
        frame_count = 0
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()
