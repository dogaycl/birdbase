import time
from ultralytics import YOLO
from florence_integration import Florence2Model

def compare_models(image_path):
    print(f"[*] Comparing Models on: {image_path}")
    
    # 1. YOLOv8
    print("\n--- YOLOv8 ---")
    try:
        yolo_model = YOLO("runs/weights/cub_v1/weights/best.onnx") # Load exported ONNX
        t0 = time.time()
        yolo_results = yolo_model(image_path, verbose=False)
        yolo_time = time.time() - t0
        print(f"[*] YOLOv8 Inference Time: {yolo_time:.4f}s")
        if len(yolo_results[0].boxes) > 0:
            box = yolo_results[0].boxes[0]
            print(f"[*] Detected: {yolo_model.names[int(box.cls)]} | Conf: {float(box.conf):.2f}")
        else:
            print("[*] Detected: None")
    except Exception as e:
        print(f"[!] YOLOv8 Error: {e}")

    # 2. Florence-2
    print("\n--- Florence-2 ---")
    try:
        # Note: Florence-2 requires downloading a multi-GB model, this is simulated.
        florence_model = Florence2Model("microsoft/Florence-2-base")
        t0 = time.time()
        florence_results = florence_model.run_inference(image_path, "<OD>")
        florence_time = time.time() - t0
        print(f"[*] Florence-2 Inference Time: {florence_time:.4f}s")
        print(f"[*] Output: {florence_results}")
    except Exception as e:
        print(f"[!] Florence-2 Error (Might need transformers/timm/flash_attn): {e}")

if __name__ == "__main__":
    # Supply a dummy image path
    # compare_models("path/to/test.jpg")
    print("Comparison module loaded.")
