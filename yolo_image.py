from ultralytics import YOLO
import cv2, time
from pathlib import Path

# model = YOLO('yolo11s.pt')
model = YOLO('yolov8n-oiv7.pt')

image_folder_str = "C:\\Users\\thoma\\Pictures\\Screenshots\\" # change to folder path of images
image_folder = os.fsencode(image_folder_str)

quit = False

for file in os.listdir(image_folder):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".png"): # put any acceptable extensions
        results = model.predict(source=image_folder_str + filename, conf=0.4)[0] # confidence threshold can be tweaked
        annotated_frame = results.plot()
        print(type(annotated_frame))
        cv2.namedWindow("YOLO predicted image", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO predicted image", annotated_frame)
        if cv2.waitKey() & 0xFF == ord('q'): # quit if q is pressed, otherwise go to next image
            quit = True;
    else:
        continue
    if quit:
        break

cv2.destroyAllWindows();
