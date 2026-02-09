from ultralytics import YOLO
import cv2, time, os
from pathlib import Path

# model = YOLO('yolo11s.pt')
model = YOLO('C:\\Users\\thoma\\Downloads\\results_yolov8n_100e\\kaggle\\working\\runs\\detect\\train\\weights\\best.pt')

# comment out if cuda is not installed/supported
#model.to("cuda")
model.fuse()

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
    loop_start_time = time.time()
    results = model.predict(frame,
                            #device=0, # comment out if cuda not installed/supported
                            verbose=False)
    annotated_frame = results[0].plot()

    frame_count +=1
    end_time = time.time()

    inference_time_ms = (end_time - loop_start_time)
    fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0

    perf_text = f"FPS: {fps:.2f} | Time/frame: {inference_time_ms:.2f} ms"
    cv2.putText(annotated_frame, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLO Real-Time Performance", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Reset timer occasionally to prevent floating point issues with very long run times
    if frame_count >= 100:
        frame_count = 0
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()
