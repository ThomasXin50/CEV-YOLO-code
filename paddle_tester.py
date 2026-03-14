import os
os.environ["FLAGS_use_mkldnn"] = "0"  # avoid oneDNN issues on CPU

import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

IMAGE_PATH = r"C:\Users\thoma\Desktop\ocr_stop.jpg"

# Initialize PaddleOCR (new API)
ocr = PaddleOCR(use_textline_orientation=True, lang="en")

def normalize(text: str):
    """Normalize OCR text for reliable comparison."""
    text = text.upper()
    text = re.sub(r"[^A-Z]", "", text)
    return text

def detect_red_regions(image: np.ndarray):
    """Return a mask of red regions in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red may wrap around HSV hue
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return mask1 | mask2

def find_octagons(mask: np.ndarray, image: np.ndarray):
    """Find candidate octagonal regions (possible stop signs)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if 7 <= len(approx) <= 9:  # roughly octagon
            x, y, w, h = cv2.boundingRect(cnt)
            crop = image[y:y+h, x:x+w].copy()
            candidates.append(crop)
    return candidates

def confirm_stop_text(region: np.ndarray):
    """Run OCR on a cropped region and confirm if it says 'STOP'."""
    # PaddleOCR v3+ expects NumPy arrays
    results = ocr.predict(region)
    for res in results:
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        for text, conf in zip(texts, scores):
            if normalize(text) == "STOP":
                print(f"STOP text detected (confidence {conf:.2f})")
                return True
    return False

def detect_stop_sign(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    mask = detect_red_regions(image)
    regions = find_octagons(mask, image)

    for region in regions:
        if confirm_stop_text(region):
            return True
    return False

if detect_stop_sign(IMAGE_PATH):
    print("STOP SIGN CONFIRMED")
else:
    print("No stop sign detected")
