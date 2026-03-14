import cv2
import numpy as np
import pytesseract
import re
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def normalize(text):
    return re.sub(r"[^A-Z]", "", text.upper())


def detect_red_regions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    return mask1 | mask2


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def preprocess_for_ocr(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def confirm_stop_text(region):

    best_conf = 0

    # check multiple rotations for tilted text
    for angle in [-15, -10, -5, 0, 5, 10, 15]:

        rotated = rotate_image(region, angle)
        processed = preprocess_for_ocr(rotated)

        data = pytesseract.image_to_data(
            processed,
            config="--psm 7",
            output_type=pytesseract.Output.DICT
        )

        for i in range(len(data["text"])):

            text = normalize(data["text"][i])
            conf = int(data["conf"][i])

            if text == "STOP":
                best_conf = max(best_conf, conf)

    return best_conf


cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    mask = detect_red_regions(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        if cv2.contourArea(cnt) < 800:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if 7 <= len(approx) <= 9:

            x, y, w, h = cv2.boundingRect(cnt)

            region = frame[y:y+h, x:x+w]

            if region.size == 0:
                continue

            confidence = confirm_stop_text(region)

            if confidence > 0:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                label = f"STOP ({confidence}%)"

                cv2.putText(
                    frame,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                print(f"STOP detected with confidence {confidence}%")

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.imshow("Stop Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
