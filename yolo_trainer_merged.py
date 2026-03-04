from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = "C:\\Users\\thoma\\Desktop\\YOLO datasets\\MERGED\\data.yaml"
# list of models to train
MODEL_LIST = [
    "yolov11n.pt",
    "yolov12n.pt",
    "yolov26n.pt",
    "yolov8n.pt"
]

# training parameters
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0

OUTPUT_ROOT = Path("path/to/training_runs")

def train_models():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for model_path in MODEL_LIST:
        print(f"\nTraining model: {model_path}")

        model = YOLO(model_path)
        model_name = Path(model_path).stem
        run_name = f"{model_name}_on_dataset"

        model.train(
            data=DATASET_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=str(OUTPUT_ROOT),
            name=run_name,
            exist_ok=True,
        )
        print(f"Finished training: {model_path}")

if __name__ == "__main__":
    train_models()
