import os
import cv2
import numpy as np
from glob import glob

folder = "/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/Gold_silver_bed_slat_interp"
output_path = "gold_slat_interp.png"

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read {video_path}")
    return frame

def parse_files(folder):
    gs_files = []
    mesh_files = []

    for f in os.listdir(folder):
        if not f.endswith(".mp4"):
            continue
        # breakpoint()

        first_val = float(f.split("_")[0])
        path = os.path.join(folder, f)

        if "_gs.mp4" in f:
            gs_files.append((first_val, path))
        elif "_mesh.mp4" in f:
            mesh_files.append((first_val, path))

    gs_files.sort(key=lambda x: x[0])
    mesh_files.sort(key=lambda x: x[0])

    return gs_files, mesh_files

def build_grid(video_list):
    frames = [get_first_frame(v[1]) for v in video_list]

    h, w, _ = frames[0].shape
    cols = len(frames)

    grid = np.zeros((h, w * cols, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        grid[:, i*w:(i+1)*w] = frame

    return grid


gs_files, mesh_files = parse_files(folder)

gs_grid = build_grid(gs_files)
mesh_grid = build_grid(mesh_files)

final = np.vstack([gs_grid, mesh_grid])

cv2.imwrite(output_path, final)

print("Saved to:", output_path)