import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import cv2


DATASET_BASE_PATH = "C:\\Users\\thoma\\Desktop\\YOLO datasets\\"
# MUST BE the detect folder
MODELS_BASE_PATH = "C:\\Users\\thoma\\Desktop\\YOLO models\\runs\\detect\\"

# These names MUST BE in order, dataset matches the model it trained
DATASET_NAMES = ["Barrels\\traffic barrel.v1i.yolov11\\data.yaml",
                 "Stop\\Stop-Sign.v2i.yolov11\\data.yaml",
                 "Tires\\Tire on Road Detection.v1i.yolov11\\data.yaml",
                 "Mannequins\\safety vest.v1i.yolov11\\data.yaml",
                 "Potholes\\potholes.v1i.yolov11\\data.yaml"]
MODELS_NAMES =  ["yolov26n barrels\\weights\\best.pt",
                 "yolov26n stop\\weights\\best.pt",
                 "yolov26n tires\\weights\\best.pt",
                 "yolov26n mannequins\\weights\\best.pt",
                 "yolov26n potholes\\weights\\best.pt"]

# Dictionary mapping dataset YAML → corresponding trained model
DATASET_MODEL_MAP = dict();
for x in range(len(DATASET_NAMES)):
    DATASET_MODEL_MAP[DATASET_BASE_PATH + DATASET_NAMES[x]] = MODELS_BASE_PATH + MODELS_NAMES[x]
# Final label list (order does not matter)
FINAL_LABEL_NAMES = [
    "barrel",
    "vest",
    "tire",
    "stop_sign",
    "pothole"
]

# Per-class confidence thresholds
PER_CLASS_CONF_THRESHOLDS = {
    "barrel": 0.30,
    "tire": 0.30,
    "stop_sign": 0.25 # lower because we will filter by text identification
}
DEFAULT_CONF_THRESHOLD = 0.35
OUTPUT_DIR = Path(DATASET_BASE_PATH + "MERGED\\")

def load_dataset_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_image_paths_from_dataset(dataset_yaml):
    image_paths = []

    for split in ["train", "val"]:
        if split in dataset_yaml:
            split_path = dataset_yaml[split]
            if os.path.isdir(split_path):
                for root, _, files in os.walk(split_path):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            image_paths.append(os.path.join(root, file))
    return image_paths


def convert_xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    return x_center, y_center, width, height


def load_original_labels(label_path):
    """
    Returns list of tuples:
    (class_id, x_center, y_center, width, height)
    """
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            labels.append((cls, x, y, w, h))
    return labels

def merge_datasets():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_out = OUTPUT_DIR / "images"
    labels_out = OUTPUT_DIR / "labels"
    images_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)

    # create name to global id mapping
    final_name_to_id = {
        name: idx for idx, name in enumerate(FINAL_LABEL_NAMES)
    }

    # Load all models
    models = {}
    for dataset_yaml, model_path in DATASET_MODEL_MAP.items():
        models[dataset_yaml] = YOLO(model_path)

    global_image_counter = 0

    # Iterate through datasets
    for dataset_yaml_path, own_model in models.items():
        
        dataset_yaml = load_dataset_yaml(dataset_yaml_path)
        image_paths = get_image_paths_from_dataset(dataset_yaml)

        print(f"\nProcessing dataset: {dataset_yaml_path}")
        print(f"Found {len(image_paths)} images")

        for img_path in image_paths:

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            # counter for image filenames in final dataset
            img_name = f"{global_image_counter:07d}_" + Path(img_path).name
            new_img_path = images_out / img_name
            new_label_path = labels_out / (Path(img_name).stem + ".txt")

            shutil.copy(img_path, new_img_path)

            merged_boxes = []

            # preserve original ground truths
            original_label_path = (
                Path(img_path).parent.parent / "labels" /
                (Path(img_path).stem + ".txt")
            )

            original_labels = load_original_labels(original_label_path)

            for cls_id, x, y, bw, bh in original_labels:
                original_names = dataset_yaml["names"]
                label_name = original_names[cls_id]
                if label_name not in final_name_to_id:
                    continue
                global_cls_id = final_name_to_id[label_name]
                merged_boxes.append((global_cls_id, x, y, bw, bh))

            for other_dataset_yaml, model in models.items():
                if other_dataset_yaml == dataset_yaml_path:
                    continue  # do NOT run model on its own dataset
                results = model(img_path)[0]
                if results.boxes is None:
                    continue

                boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()

                for cls_id, box, conf in zip(class_ids, boxes_xyxy, confidences):
                    label_name = model.names[cls_id]
                    if label_name not in final_name_to_id:
                        continue
                    threshold = PER_CLASS_CONF_THRESHOLDS.get(
                        label_name,
                        DEFAULT_CONF_THRESHOLD
                    )
                    if conf < threshold:
                        continue
                    global_cls_id = final_name_to_id[label_name]

                    x_center, y_center, bw, bh = convert_xyxy_to_yolo(box, w, h)

                    merged_boxes.append(
                        (global_cls_id, x_center, y_center, bw, bh)
                    )

            # write
            with open(new_label_path, "w") as f:
                for cls_id, x, y, bw, bh in merged_boxes:
                    f.write(f"{cls_id} {x} {y} {bw} {bh}\n")

            global_image_counter += 1

    # write final data.yaml
    merged_yaml = {
        "path": str(OUTPUT_DIR),
        "train": "images",
        "val": "images",
        "names": FINAL_LABEL_NAMES,
        "nc": len(FINAL_LABEL_NAMES),
    }

    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(merged_yaml, f)

    print("\nMerged dataset complete.")

if __name__ == "__main__":
    merge_datasets()
