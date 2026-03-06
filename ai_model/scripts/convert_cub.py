import os
import shutil
import cv2
from pathlib import Path

# Paths
CUB_DIR = Path("../data/CUB_200_2011/CUB_200_2011")
YOLO_DIR = Path("../data/cub_yolo")

IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

def read_txt_to_dict(filepath, split_char=' '):
    d = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(split_char)
            if len(parts) >= 2:
                d[int(parts[0])] = parts[1:] if len(parts) > 2 else parts[1]
    return d

def read_bboxes(filepath):
    d = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 5:
                # id, x, y, width, height (all floats except id)
                d[int(parts[0])] = [float(x) for x in parts[1:]]
    return d

def main():
    print("[*] Starting conversion of CUB-200-2011 to YOLO format...")
    
    # Read metadata
    images = read_txt_to_dict(CUB_DIR / "images.txt")
    bboxes = read_bboxes(CUB_DIR / "bounding_boxes.txt")
    splits = read_txt_to_dict(CUB_DIR / "train_test_split.txt")
    labels = read_txt_to_dict(CUB_DIR / "image_class_labels.txt")
    classes = read_txt_to_dict(CUB_DIR / "classes.txt")
    
    # Create directories
    for phase in ["train", "val"]:
        (IMAGES_DIR / phase).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / phase).mkdir(parents=True, exist_ok=True)
        
    # Convert and copy
    count_train = 0
    count_val = 0
    
    for img_id, rel_img_path in images.items():
        is_train = int(splits[img_id]) == 1
        phase = "train" if is_train else "val"
        
        # YOLO class index is 0-based
        class_idx = int(labels[img_id]) - 1
        
        # Original bbox (x, y, w, h)
        x_min, y_min, w, h = bboxes[img_id]
        
        # Read image to get width and height for normalization
        src_img_path = CUB_DIR / "images" / rel_img_path
        if not src_img_path.exists():
            print(f"[!] Warning: Image not found -> {src_img_path}")
            continue
            
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Calculate YOLO normalized bbox
        x_center = x_min + w / 2.0
        y_center = y_min + h / 2.0
        
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # Copy image
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
            
    print(f"[*] Conversion completed. Train images: {count_train}, Val images: {count_val}")
    
    # Generate cub_dataset.yaml
    yaml_path = YOLO_DIR / "cub_dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write("path: ../data/cub_yolo\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/val\n\n")
        
        f.write(f"nc: {len(classes)}\n")
        f.write("names: [\n")
        for cls_id in sorted(classes.keys()):
            # e.g. "001.Black_footed_Albatross" -> "Black footed Albatross"
            name = classes[cls_id].split('.', 1)[1].replace('_', ' ')
            f.write(f"  '{name}',\n")
        f.write("]\n")
        
    print(f"[*] Generated YOLO config at {yaml_path}")

if __name__ == "__main__":
    main()
