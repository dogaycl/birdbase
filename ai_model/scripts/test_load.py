import torch
from ultralytics import YOLO

zip_path = "../weights/best.pt.zip"
renamed_path = "../weights/best.pt"

try:
    print(f"[*] Trying to load {renamed_path} directly...")
    model = YOLO(renamed_path)
    print("Success! YOLO loaded it.")
except Exception as e:
    print(f"Error loading renamed zip: {e}")
