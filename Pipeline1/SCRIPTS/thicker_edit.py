import numpy as np
import os
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Environment settings
# os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

# Load pipeline once
pipeline = TrellisImageTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-image-large")
pipeline.cuda()

# Load images once
image1 = Image.open("/mnt/data/srihari/my_TRELLIS/panda.png")
image2 = Image.open("/mnt/data/srihari/my_TRELLIS/husky.png")


# # Thickener
# image1 = Image.open("slim.png")
# image2 = Image.open("thick.png")







TITLE = "Panda_husky_slerp_unit_vector"
OUT_FOLDER = f"ALL_OUTPUTS/{TITLE}"
os.makedirs(OUT_FOLDER , exist_ok = True)
# Alpha values from 0 to 1 with step 0.1 (inclusive)
alphas = np.linspace(0,1.1,10)

for alpha in alphas:
    alpha = float(alpha)
    print(f"ALPHA = {alpha} , {type(alpha)}")

    NAME = f"{alpha:.2f}"

    outputs = pipeline.run_edit_sparse(
        image1=image1,
        image2=image2,
        seed=1,
        alpha=alpha
    )

    # Render Gaussian
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{OUT_FOLDER}/{NAME}_gs.mp4", video, fps=30)

    # Render Mesh
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{OUT_FOLDER}/{NAME}_mesh.mp4", video, fps=30)

    print(f"Finished alpha = {alpha}")



# GRID MAKING : 

import os
import cv2
import numpy as np
from glob import glob

folder = OUT_FOLDER
output_path = f"{TITLE}.png"


import cv2

def get_mid_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read frame count for {video_path}")
    
    # Calculate mid index (integer division)
    mid_frame_index = total_frames // 2
    
    # Set the playhead to the middle
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read middle frame at index {mid_frame_index}")
        
    return frame
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
    frames = [get_mid_frame(v[1]) for v in video_list]

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
    
