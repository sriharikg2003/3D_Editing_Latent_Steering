"""


Generate an object 
"""

import os
import torch

os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from PIL import Image
import imageio

pipeline = TrellisImageTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-image-large")
pipeline.cuda()

# image = Image.open("/mnt/data/srihari/my_TRELLIS/assets/sneaker.webp").convert("RGB")

# output = pipeline.run(
#     image,
#     seed=42,
#     sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
#     slat_sampler_params={"steps": 12, "cfg_strength": 7.5},
# )

# # Save render
# video = render_utils.render_video(output['gaussian'][0])['color']
# imageio.mimsave("sneaker_3d.mp4", video, fps=30)
# print("Done")

# # Also get raw SLAT — we need this
# cond = pipeline.get_cond([image])
# torch.manual_seed(42)
# coords = pipeline.sample_sparse_structure(cond, 1, {"steps": 12, "cfg_strength": 7.5})
# slat = pipeline.sample_slat(cond, coords, {"steps": 12, "cfg_strength": 7.5})
# torch.save({'coords': slat.coords, 'feats': slat.feats}, "sneaker_slat.pt")

# print("SLAT coords:", slat.coords.shape)
# print("SLAT feats:", slat.feats.shape)


"""
Squishing experiment
"""
import torch
import trellis.modules.sparse as sp
from trellis.utils import render_utils
import imageio

# Load saved SLAT
data = torch.load("sneaker_slat.pt")
coords = data['coords']  # [11164, 4]
feats  = data['feats']   # [11164, 8]

# Check coordinate ranges to find vertical axis
# print("axis 1 range:", coords[:, 1].min().item(), coords[:, 1].max().item())
# print("axis 2 range:", coords[:, 2].min().item(), coords[:, 2].max().item())  
# print("axis 3 range:", coords[:, 3].min().item(), coords[:, 3].max().item())



# slat_orig = sp.SparseTensor(feats=feats, coords=coords)

# # Squish: compress Y axis (column 2 of coords = z in voxel space)
# # Try each axis to see which one is vertical
# frames = []

# for alpha in list(torch.linspace(0, 1, 15)) + list(torch.linspace(1, 0, 15)):
#     alpha = alpha.item()
    
#     new_coords = coords.clone().float()
#     new_coords[:, 2] = coords[:, 2] * (1 - 0.5 * alpha)  # squish by 50% max
#     new_coords = new_coords.int()
    
#     slat_interp = sp.SparseTensor(feats=feats, coords=new_coords)
    
#     # Decode and render one frame
#     decoded = pipeline.decode_slat(slat_interp, formats=['gaussian'])
#     frame = render_utils.render_video(decoded['gaussian'][0], 
#                                        num_frames=1)['color'][0]
#     frames.append(frame)

# imageio.mimsave("bounce.mp4", frames, fps=15)
# print("Done — check bounce.mp4")
import torch
import trellis.modules.sparse as sp
from trellis.utils import render_utils
import imageio

data = torch.load("sneaker_slat.pt")
coords = data['coords']
feats  = data['feats']

y_min = coords[:, 2].float().min()
y_max = coords[:, 2].float().max()
y_mid = (y_min + y_max) / 2

# Step 1: pre-render all 20 bounce states as full 60-frame rotation videos
print("Pre-rendering bounce states...")
rotation_videos = []

alphas = list(torch.linspace(0, 1, 10)) + list(torch.linspace(1, 0, 10))

for i, alpha in enumerate(alphas):
    alpha = alpha.item()
    scale = 1 - 0.4 * alpha

    new_coords = coords.clone().float()
    new_coords[:, 2] = (coords[:, 2].float() - y_mid) * scale + y_mid
    new_coords = new_coords.int()

    slat_interp = sp.SparseTensor(feats=feats, coords=new_coords)
    decoded = pipeline.decode_slat(slat_interp, formats=['gaussian'])
    video = render_utils.render_video(decoded['gaussian'][0], num_frames=60)['color']
    rotation_videos.append(video)  # list of [60, H, W, 3]
    print(f"  state {i+1}/20 done")

# Step 2: interleave — for each rotation frame, pick the bounce state
# Full animation = 60 rotation frames, bounce cycles 3 times
final_frames = []
n_bounce_cycles = 3
total_frames = 60

for rot_frame in range(total_frames):
    # Which bounce state are we in?
    bounce_phase = (rot_frame / total_frames * n_bounce_cycles) % 1.0
    bounce_idx = int(bounce_phase * len(alphas)) % len(alphas)
    
    final_frames.append(rotation_videos[bounce_idx][rot_frame])

imageio.mimsave("sneaker_bounce_rotate.mp4", final_frames, fps=24)
print("Done — sneaker_bounce_rotate.mp4")