import os
from ultralytics import YOLO

def train_model():
    print("[*] Initializing YOLOv8 nano model...")
    # Load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt") 

    # Train the model
    print("[*] Starting training process...")
    results = model.train(
        data="../data/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="../weights",
        name="birdbase_v1",
        device="cpu", # Change to "cuda" if GPU is available
        val=True       # Validate during training
    )

    print("[*] Training completed. Best model saved in ../weights/birdbase_v1/weights/best.pt")

if __name__ == "__main__":
    train_model()
