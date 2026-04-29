import cv2
import os
import glob
import matplotlib.pyplot as plt


folder_path = f'/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/SLAT_INTERP_diff_hair_color_girl'
NAME =folder_path.split('/')[-1]

video_files = sorted(glob.glob(os.path.join(folder_path, '*.mp4')))

frames = []

for vid_path in video_files:
    vid_name = os.path.basename(vid_path)
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        continue
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((vid_name, frame_rgb))
    
    cap.release()

if not frames:
    print("No frames were captured. Check the folder path.")
else:
    num_videos = len(frames)
    fig, axes = plt.subplots(1, num_videos, figsize=(4 * num_videos, 6))
    
    if num_videos == 1:
        axes = [axes]

    for i, (name, frame) in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(name, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(folder_path, f'{NAME}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grid saved to: {save_path}")