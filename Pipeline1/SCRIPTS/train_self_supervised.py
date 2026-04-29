import numpy as np
import torch
from pathlib import Path
import trellis.modules.sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
import imageio
import os

os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

LATENT_DIR = "/val_data/datasets/HSSD_FROM_55/HSSD/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
files = sorted(Path(LATENT_DIR).glob("*.npz"))

# Load two objects
A = np.load(str(files[0]))
B = np.load(str(files[1]))

coords_A = A['coords']  # [Na, 3] uint8
feats_A  = A['feats']   # [Na, 8] float32
coords_B = B['coords']  # [Nb, 3] uint8
feats_B  = B['feats']   # [Nb, 8] float32

# Build lookup dict: position → feature
dict_A = {tuple(c): f for c, f in zip(coords_A, feats_A)}
dict_B = {tuple(c): f for c, f in zip(coords_B, feats_B)}

all_positions = set(dict_A.keys()) | set(dict_B.keys())

def interpolate_slat(dict_A, dict_B, all_positions, alpha, seed=42):
    np.random.seed(seed)
    out_coords = []
    out_feats  = []

    for pos in all_positions:
        in_A = pos in dict_A
        in_B = pos in dict_B

        if in_A and in_B:
            # Shared: always keep, interpolate features
            feat = (1 - alpha) * dict_A[pos] + alpha * dict_B[pos]
            out_coords.append(pos)
            out_feats.append(feat)

        elif in_A:
            # Only in A: keep with prob (1-alpha)
            if np.random.rand() < (1 - alpha):
                out_coords.append(pos)
                out_feats.append(dict_A[pos])

        else:
            # Only in B: keep with prob alpha
            if np.random.rand() < alpha:
                out_coords.append(pos)
                out_feats.append(dict_B[pos])

    coords = np.array(out_coords, dtype=np.int32)  # [N, 3]
    feats  = np.array(out_feats,  dtype=np.float32) # [N, 8]
    return coords, feats


def make_sparse_tensor(coords_np, feats_np):
    coords = torch.from_numpy(coords_np)
    feats  = torch.from_numpy(feats_np)
    batch  = torch.zeros(coords.shape[0], 1, dtype=torch.int32)
    coords = torch.cat([batch, coords], dim=1)
    return sp.SparseTensor(feats=feats.cuda(), coords=coords.cuda())


# Load pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "/datafrom_146/srihari/MODELS/TRELLIS-image-large"
)
pipeline.cuda()

# Generate interpolations
os.makedirs("CROSS_INTERP", exist_ok=True)
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

for alpha in alphas:
    print(f"Interpolating α={alpha}...")
    coords, feats = interpolate_slat(dict_A, dict_B, all_positions, alpha)
    print(f"  Voxels: {len(coords)}")

    slat = make_sparse_tensor(coords, feats)
    decoded = pipeline.decode_slat(slat, formats=['gaussian'])
    video = render_utils.render_video(
        decoded['gaussian'][0], num_frames=60
    )['color']
    imageio.mimsave(f"CROSS_INTERP/alpha_{alpha:.2f}.mp4", video, fps=30)
    print(f"  Saved alpha_{alpha:.2f}.mp4")

print("Done. Check CROSS_INTERP/")