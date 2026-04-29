import os
import cv2
import numpy as np
from glob import glob

ROOT = "/mnt/data/srihari/my_TRELLIS/patched_steering_chair"
OUTPUT_PATH = "crop_alpha_chair.png"

CELL_SIZE = 256
FONT = cv2.FONT_HERSHEY_SIMPLEX

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed: {video_path}")

    return cv2.resize(frame, (CELL_SIZE, CELL_SIZE))


def extract_alpha(fp):
    name = os.path.basename(fp)
    return float(name.split("_")[1].replace(".mp4", ""))


def extract_crop_name(path):
    return os.path.basename(path)  # crop_04 etc


# ---- Load structure ----
crop_dirs = sorted(glob(os.path.join(ROOT, "crop_*")))
crop_names = [extract_crop_name(c) for c in crop_dirs]

alpha_files = sorted(glob(os.path.join(crop_dirs[0], "*.mp4")), key=extract_alpha)
alphas = [extract_alpha(f) for f in alpha_files]

num_rows = len(alphas)
num_cols = len(crop_dirs)

# ---- Create canvas with padding for labels ----
LEFT_PAD = 120
TOP_PAD = 80

grid_h = num_rows * CELL_SIZE
grid_w = num_cols * CELL_SIZE

canvas = np.ones(
    (grid_h + TOP_PAD, grid_w + LEFT_PAD, 3),
    dtype=np.uint8
) * 255  # white background

# ---- Fill images ----
for r, alpha in enumerate(alphas):
    for c, crop_dir in enumerate(crop_dirs):
        files = sorted(glob(os.path.join(crop_dir, "*.mp4")), key=extract_alpha)
        frame = extract_middle_frame(files[r])

        y = TOP_PAD + r * CELL_SIZE
        x = LEFT_PAD + c * CELL_SIZE

        canvas[y:y+CELL_SIZE, x:x+CELL_SIZE] = frame

# ---- Draw column headers (crop names) ----
for c, name in enumerate(crop_names):
    x = LEFT_PAD + c * CELL_SIZE + 20
    y = 40
    cv2.putText(canvas, name, (x, y), FONT, 0.6, (0, 0, 0), 2)

# ---- Draw row headers (alpha values) ----
for r, alpha in enumerate(alphas):
    y = TOP_PAD + r * CELL_SIZE + CELL_SIZE // 2
    text = f"{alpha:.3f}"
    cv2.putText(canvas, text, (10, y), FONT, 0.6, (0, 0, 0), 2)

# ---- Save ----
cv2.imwrite(OUTPUT_PATH, canvas)
print(f"Saved labeled grid → {OUTPUT_PATH}")