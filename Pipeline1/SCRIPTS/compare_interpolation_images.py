import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

def get_middle_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

pairs = ["pair_0", "pair_1", "pair_2"]
steps = [f"step_{i:02d}.mp4" for i in range(10)]

for pair in pairs:
    linear_frames = [get_middle_frame(f"LINEAR_INTERP/{pair}/{s}") for s in steps]
    slerp_frames  = [get_middle_frame(f"SLERP_INTERP/{pair}/{s}")  for s in steps]

    fig, axes = plt.subplots(2, 10, figsize=(30, 6))
    fig.suptitle(f"{pair} — Top: LERP   Bottom: SLERP", fontsize=14)

    for j in range(10):
        axes[0, j].imshow(linear_frames[j])
        axes[0, j].set_title(f"α={j/9:.1f}", fontsize=8)
        axes[0, j].axis('off')

        axes[1, j].imshow(slerp_frames[j])
        axes[1, j].set_title(f"α={j/9:.1f}", fontsize=8)
        axes[1, j].axis('off')

    axes[0, 0].set_ylabel("LERP", fontsize=10)
    axes[1, 0].set_ylabel("SLERP", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"comparison_{pair}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison_{pair}.png")