import os
from ultralytics import YOLO

def train_nabirds():
    print("[*] Initializing YOLOv8 nano model for NAbirds...")
    # İhtiyaca göre yolov8s.pt, yolov8m.pt gibi daha büyük modeller kullanılabilir
    model = YOLO("yolov8n.pt") 

    print("[*] Starting training process for NAbirds...")
    # Eğitim parametrelerini (epochs, batch, imgsz vb.) donanımınıza göre değiştirebilirsiniz
    results = model.train(
        data="../data/nabirds_yolo/nabirds_dataset.yaml",
        epochs=50, # Gerçek eğitim için 50-100+ olarak ayarlayın
        imgsz=640,
        batch=16,
        project="../weights",
        name="nabirds_v1",
        device="cpu", # Apple Silicon (M1/M2/M3) için "mps", Nvidia GPU için "cuda" yapabilirsiniz
        val=True
    )

    print("[*] Training completed. Best model saved in ../weights/nabirds_v1/weights/best.pt")

if __name__ == "__main__":
    train_nabirds()
