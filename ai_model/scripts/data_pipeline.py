import os
import cv2
import glob
import shutil
import random
import numpy as np
from pathlib import Path

# Paths
RAW_DATA_DIR = Path("../data/raw")
PROCESSED_DATA_DIR = Path("../data/processed")
SPLIT_DATA_DIR = Path("../data/splits")

def setup_directories():
    """Create necessary directories if they don't exist."""
    print("[*] Setting up directories...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def collect_data():
    """
    Placeholder for data collection logic.
    Could be Web Scraping, Kaggle API download, etc.
    """
    print("[*] Running data collection (Placeholder)...")
    # E.g. usage of kaggle.api.dataset_download_files()
    pass

def clean_data(input_dir, output_dir):
    """
    Filter corrupt or unreadable images.
    """
    print(f"[*] Cleaning data from {input_dir} to {output_dir}...")
    valid_extensions = {".jpg", ".jpeg", ".png"}
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count_cleaned = 0
    
    for filepath in input_path.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
            # Try to read image to ensure it's not corrupt
            img = cv2.imread(str(filepath))
            if img is not None:
                # Save to processed
                class_name = filepath.parent.name
                class_dir = output_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(filepath, class_dir / filepath.name)
                count_cleaned += 1
            else:
                print(f"[!] Warning: Corrupt image ignored -> {filepath}")
    
    print(f"[*] Cleaned {count_cleaned} valid images.")

def augment_image(image):
    """
    Apply augmentations: rotate, flip, brightness
    Return a list of augmented images.
    """
    augmented = []
    
    # 1. Original
    augmented.append(image)
    
    # 2. Horizontal Flip
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    
    # 3. Rotate 15 degrees
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    augmented.append(rotated)
    
    # 4. Brightness adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 30)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    augmented.append(bright)
    
    return augmented

def augment_dataset(input_dir):
    """
    Augment images in the processed directory in place.
    """
    print(f"[*] Augmenting dataset in {input_dir}...")
    input_path = Path(input_dir)
    count_aug = 0
    
    for filepath in list(input_path.rglob("*.*")):
        if filepath.is_file():
            img = cv2.imread(str(filepath))
            if img is not None:
                augs = augment_image(img)
                # Skip index 0 (original)
                for i, aug_img in enumerate(augs[1:], start=1):
                    new_name = f"{filepath.stem}_aug{i}{filepath.suffix}"
                    new_path = filepath.parent / new_name
                    cv2.imwrite(str(new_path), aug_img)
                    count_aug += 1
    
    print(f"[*] Generated {count_aug} augmented images.")

def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Split the dataset into train, val, and test sets.
    """
    print(f"[*] Splitting dataset into train/val/test...")
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    for _dir in [train_dir, val_dir, test_dir]:
        _dir.mkdir(parents=True, exist_ok=True)
        
    classes = [d for d in input_path.iterdir() if d.is_dir()]
    
    for cls_path in classes:
        cls_name = cls_path.name
        
        images = list(cls_path.glob("*.*"))
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        for phase, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            phase_dir = output_path / phase / cls_name
            phase_dir.mkdir(parents=True, exist_ok=True)
            for img_path in img_list:
                shutil.copy2(img_path, phase_dir / img_path.name)
                
    print(f"[*] Split complete. Outputs saved in {output_dir}")

if __name__ == "__main__":
    setup_directories()
    collect_data()
    clean_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    augment_dataset(PROCESSED_DATA_DIR)
    split_dataset(PROCESSED_DATA_DIR, SPLIT_DATA_DIR)
    print("[*] Data Pipeline Completed.")
