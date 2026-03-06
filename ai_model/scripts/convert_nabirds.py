import os
import shutil
import cv2
import random
from pathlib import Path

# Paths
NABIRDS_DIR = Path("../data/NAbirds/raw")
YOLO_DIR = Path("../data/nabirds_yolo")

IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

def read_txt_to_dict(filepath, split_char=' '):
    d = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(split_char, 1) # Support spaces in second part
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d

def read_txt_to_list(filepath):
    d = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1:]
    return d

def main():
    print("[*] Starting conversion of NAbirds to YOLO format...")
    
    # Check if raw directory exists
    if not NABIRDS_DIR.exists():
        print(f"[!] Error: Raw data directory not found at {NABIRDS_DIR}")
        print("    Please ensure you have unzipped archive.zip inside data/NAbirds/raw/")
        return

    # Read metadata
    images_file = NABIRDS_DIR / "images.txt"
    bboxes_file = NABIRDS_DIR / "bounding_boxes.txt"
    labels_file = NABIRDS_DIR / "image_class_labels.txt"
    classes_file = NABIRDS_DIR / "classes.txt"
    train_test_file = NABIRDS_DIR / "train_test_split.txt"

    images = read_txt_to_dict(images_file)
    bboxes = read_txt_to_list(bboxes_file)
    labels = read_txt_to_dict(labels_file)
    classes = read_txt_to_dict(classes_file)

    # Some NAbirds releases might skip train_test_split.txt
    splits = {}
    if train_test_file.exists():
        splits = read_txt_to_dict(train_test_file)
        print("[*] Found existing train_test_split.txt")
    else:
        print("[*] No train_test_split.txt found. Generating an 80/20 train/val split dynamically...")
        all_ids = list(images.keys())
        random.seed(42)
        random.shuffle(all_ids)
        split_idx = int(len(all_ids) * 0.8)
        
        for img_id in all_ids[:split_idx]:
            splits[img_id] = "1"  # Train
        for img_id in all_ids[split_idx:]:
            splits[img_id] = "0"  # Val

    # Create directories
    for phase in ["train", "val"]:
        (IMAGES_DIR / phase).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / phase).mkdir(parents=True, exist_ok=True)
        
    print("[*] Processing images and generating bounding box labels. This may take a few minutes...")
    
    # Map class_ids to 0-based sequential integers for YOLO
    # YOLO requires class indices to be contiguous starting from 0 to N-1
    # NAbirds class IDs usually map from a set of IDs that might skip numbers
    sorted_class_ids = sorted(classes.keys(), key=lambda x: int(x))
    class_id_to_yolo_idx = {cls_id: i for i, cls_id in enumerate(sorted_class_ids)}
    
    count_train = 0
    count_val = 0
    missing_images = 0
    
    for img_id, rel_img_path in images.items():
        is_train = int(splits.get(img_id, '0')) == 1
        phase = "train" if is_train else "val"
        
        # Original class_id
        original_class_id = labels.get(img_id)
        if original_class_id not in class_id_to_yolo_idx:
            continue
            
        # YOLO 0-based index
        class_idx = class_id_to_yolo_idx[original_class_id]
        
        # Original bbox (x, y, w, h)
        bbox_data = bboxes.get(img_id)
        if not bbox_data or len(bbox_data) < 4:
            continue
            
        x_min, y_min, w, h = [float(x) for x in bbox_data[:4]]
        
        # Read image to get width and height for normalization
        src_img_path = NABIRDS_DIR / "images" / rel_img_path
        if not src_img_path.exists():
            missing_images += 1
            continue
            
        img = cv2.imread(str(src_img_path))
        if img is None:
            missing_images += 1
            print(f"[!] Warning: Cannot read image -> {src_img_path}")
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Ensure bbox doesn't exceed image boundaries
        w = min(w, img_w - x_min)
        h = min(h, img_h - y_min)
        
        # Calculate YOLO normalized bbox
        x_center = x_min + w / 2.0
        y_center = y_min + h / 2.0
        
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # Copy image
        # Using format: img_id.jpg to prevent collisions if image names are duplicate across folders
        dst_img_path = IMAGES_DIR / phase / f"{img_id}.jpg"
        shutil.copy2(src_img_path, dst_img_path)
        
        # Write YOLO label
        dst_label_path = LABELS_DIR / phase / f"{img_id}.txt"
        with open(dst_label_path, 'w') as f:
            f.write(f"{class_idx} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
        if is_train:
            count_train += 1
        else:
            count_val += 1
            
        if (count_train + count_val) % 5000 == 0:
            print(f"    - Processed {count_train + count_val} images so far...")
            
    print(f"[*] Conversion completed!")
    print(f"    - Train images: {count_train}")
    print(f"    - Val images: {count_val}")
    if missing_images > 0:
        print(f"    - Missing/Failed images: {missing_images}")
    
    # Generate nabirds_dataset.yaml
    yaml_path = YOLO_DIR / "nabirds_dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write("path: ../data/nabirds_yolo\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/val\n\n")
        
        f.write(f"nc: {len(sorted_class_ids)}\n")
        f.write("names: [\n")
        for cls_id in sorted_class_ids:
            name = classes[cls_id].replace("'", "''") # escape quotes for YAML via doubling
            f.write(f"  '{name}',\n")
        f.write("]\n")
        
    print(f"[*] Generated YOLO config at {yaml_path}")

if __name__ == "__main__":
    main()
