import os
import shutil
import random
from pathlib import Path

# Customize these paths
original_data = Path("../dataset/animal-10/raw-img")  # raw folders from Animals-10
output_dir = Path("../dataset/animals10")  # where to create train/val folders

train_split = 0.8  # 80% train, 20% val

# Create target folders
for phase in ['train', 'val']:
    for class_dir in os.listdir(original_data):
        os.makedirs(output_dir / phase / class_dir, exist_ok=True)

# Split images
for class_dir in os.listdir(original_data):
    files = os.listdir(original_data / class_dir)
    random.shuffle(files)
    
    split_point = int(len(files) * train_split)
    train_files = files[:split_point]
    val_files = files[split_point:]

    for f in train_files:
        shutil.copy(original_data / class_dir / f, output_dir / "train" / class_dir / f)

    for f in val_files:
        shutil.copy(original_data / class_dir / f, output_dir / "val" / class_dir / f)

print("✅ Dataset split complete!")