import os
import random
import shutil
from pathlib import Path

src_root = "/val_data/datasets/HSSD_FROM_55/HSSD/renders"
dst_root = "/datafrom_146/srihari/my_TRELLIS/assets/SAMPLED_FROM_HSSD"

os.makedirs(dst_root, exist_ok=True)

for folder in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder)

    if not os.path.isdir(folder_path):
        continue

    # collect image files
    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        print(f"No images in {folder}")
        continue

    # pick random image
    img = random.choice(images)

    src_img = os.path.join(folder_path, img)
    dst_img = os.path.join(dst_root, f"{folder}.png")

    shutil.copy(src_img, dst_img)

    print(f"Saved {dst_img}")