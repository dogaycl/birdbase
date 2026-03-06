import os
from ultralytics import YOLO

def train_cub():
    print("[*] Initializing YOLOv8 nano model for CUB-200-2011...")
    model = YOLO("yolov8n.pt") 

    print("[*] Starting training process... (This might take a while on a CPU!)")
    # Reducing epochs to 5 for demonstration purposes. Real training requires 50-100+ epochs on a GPU.
    results = model.train(
        data="../data/cub_yolo/cub_dataset.yaml",
        epochs=1,
        imgsz=640,
        batch=16,
        project="../weights",
        name="cub_v1",
        device="cpu", # Change to "cuda" if GPU is available
        val=True
    )

    print("[*] Training completed. Best model saved in ../weights/cub_v1/weights/best.pt")

if __name__ == "__main__":
    train_cub()
