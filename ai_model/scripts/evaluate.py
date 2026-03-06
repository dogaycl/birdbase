import os
from ultralytics import YOLO

def evaluate_model():
    print("[*] Loading best trained model...")
    weights_path = "../weights/birdbase_v1/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"[!] Error: Model weights not found at {weights_path}")
        return

    model = YOLO(weights_path)

    # Evaluate the model on the validation set
    print("[*] Running evaluation...")
    metrics = model.val() # Evaluates using the parameters in dataset.yaml

    print(f"[*] mAP50-95: {metrics.box.map}")        # map50-95
    print(f"[*] mAP50: {metrics.box.map50}")         # map50
    print(f"[*] mAP75: {metrics.box.map75}")         # map75
    print(f"[*] Precision: {metrics.box.mp}")        # Precision
    print(f"[*] Recall: {metrics.box.mr}")           # Recall
    print("[*] Evaluation Completed.")

if __name__ == "__main__":
    evaluate_model()
