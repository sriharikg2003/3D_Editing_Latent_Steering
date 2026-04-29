"""
inference.py
------------
Load trained model, apply deformation to a new object,
correct features with network, decode and render.

Usage:
    python inference.py --npz path/to/object.npz --deform squish --strength 0.6
"""

import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

import torch
import torch.nn as nn
import numpy as np
import imageio
import argparse
from pathlib import Path

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
import trellis.modules.sparse as sp

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/mnt/data/srihari/MODELS/TRELLIS-image-large"
CKPT_PATH   = "CHECKPOINTS/best.pt"
FEAT_DIM    = 8
COORD_DIM   = 3
HIDDEN_DIM  = 256
N_HEADS     = 4
N_LAYERS    = 4
# ──────────────────────────────────────────────────────────────────────────────


# ── Model (same as train script) ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        sa, _ = self.self_attn(q, q, q)
        q = self.norm1(q + sa)
        ca, _ = self.cross_attn(q, kv, kv)
        q = self.norm2(q + ca)
        q = self.norm3(q + self.ff(q))
        return q


class DeformationFeatureUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_encoder = nn.Sequential(
            nn.Linear(COORD_DIM + FEAT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.tgt_encoder = nn.Sequential(
            nn.Linear(COORD_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(HIDDEN_DIM, N_HEADS)
            for _ in range(N_LAYERS)
        ])
        self.out_proj = nn.Linear(HIDDEN_DIM, FEAT_DIM)

    def forward(self, orig_coords, orig_feats, tgt_coords):
        src = self.src_encoder(
            torch.cat([orig_coords, orig_feats], dim=-1)
        ).unsqueeze(0)
        tgt = self.tgt_encoder(tgt_coords).unsqueeze(0)
        for layer in self.layers:
            tgt = layer(tgt, src)
        return self.out_proj(tgt.squeeze(0))


# ── Deformations ──────────────────────────────────────────────────────────────

def squish_coords(coords, scale):
    coords = coords.clone().float()
    center_y = coords[:, 1].mean()
    coords[:, 1] = (coords[:, 1] - center_y) * scale + center_y
    return coords

def twist_coords(coords, angle_deg):
    coords = coords.clone().float()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    y_range = (y_max - y_min).clamp(min=1e-8)
    t = (coords[:, 1] - y_min) / y_range
    angles = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32) * t)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    x, z = coords[:, 0].clone(), coords[:, 2].clone()
    coords[:, 0] =  cos_a * x + sin_a * z
    coords[:, 2] = -sin_a * x + cos_a * z
    return coords

def taper_coords(coords, top_scale):
    coords = coords.clone().float()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    y_range = (y_max - y_min).clamp(min=1e-8)
    t = (coords[:, 1] - y_min) / y_range
    scale = 1.0 - (1.0 - top_scale) * t
    cx = coords[:, 0].mean()
    cz = coords[:, 2].mean()
    coords[:, 0] = cx + (coords[:, 0] - cx) * scale
    coords[:, 2] = cz + (coords[:, 2] - cz) * scale
    return coords


# ── Inference ─────────────────────────────────────────────────────────────────

def make_slat_tensor(coords_np, feats_np):
    """Convert numpy arrays to TRELLIS SparseTensor format."""
    coords = torch.from_numpy(coords_np.astype(np.int32))
    feats  = torch.from_numpy(feats_np.astype(np.float32))
    # Add batch dim to coords
    batch  = torch.zeros(coords.shape[0], 1, dtype=torch.int32)
    coords = torch.cat([batch, coords], dim=1)
    return sp.SparseTensor(feats=feats.cuda(), coords=coords.cuda())


def render_comparison(pipeline, slat_orig, slat_naive, slat_corrected, out_path):
    """Render 3 objects side by side: original, naive deform, corrected."""
    frames = []
    for slat, label in [
        (slat_orig,      "Original"),
        (slat_naive,     "Naive (no correction)"),
        (slat_corrected, "Corrected (ours)"),
    ]:
        decoded = pipeline.decode_slat(slat, formats=['gaussian'])
        video   = render_utils.render_video(
            decoded['gaussian'][0], num_frames=60
        )['color']
        frames.append((video, label))
        print(f"  Rendered: {label}")

    # Save individual videos
    stem = Path(out_path).stem
    for (video, label) in frames:
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        imageio.mimsave(f"{stem}_{safe_label}.mp4", video, fps=30)

    print(f"Saved videos to {stem}_*.mp4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',      type=str, required=True,
                        help='Path to .npz SLAT file')
    parser.add_argument('--deform',   type=str, default='squish',
                        choices=['squish', 'twist', 'taper'])
    parser.add_argument('--strength', type=float, default=0.6,
                        help='Deformation strength (squish scale, twist degrees, taper scale)')
    parser.add_argument('--out',      type=str, default='result')
    args = parser.parse_args()

    # Load TRELLIS pipeline for decoding
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_PATH)
    pipeline.cuda()

    # Load trained model
    print("Loading trained model...")
    model = DeformationFeatureUpdater().cuda()
    ckpt  = torch.load(CKPT_PATH, map_location='cuda')
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.5f})")

    # Load SLAT
    data       = np.load(args.npz)
    orig_feats = data['feats'].astype(np.float32)   # [V, 8]
    orig_coords = data['coords'].astype(np.float32)  # [V, 3]

    print(f"Object: {args.npz}  voxels={orig_coords.shape[0]}")

    # Normalize coords
    # Subsample to MAX_VOXELS to avoid OOM
    MAX_VOXELS = 4096
    V = orig_coords.shape[0]
    if V > MAX_VOXELS:
        idx = np.random.choice(V, MAX_VOXELS, replace=False)
        orig_coords = orig_coords[idx]
        orig_feats  = orig_feats[idx]
        print(f"  Subsampled {V} → {MAX_VOXELS} voxels")

    # Normalize coords
    coords_norm = torch.from_numpy(orig_coords / 63.0).cuda()
    feats_t     = torch.from_numpy(orig_feats).cuda()

    # Apply deformation to coords
    if args.deform == 'squish':
        def_coords_norm = squish_coords(coords_norm, args.strength)
    elif args.deform == 'twist':
        def_coords_norm = twist_coords(coords_norm, args.strength)
    else:
        def_coords_norm = taper_coords(coords_norm, args.strength)

    # Naive deform: move coords, keep original features (broken baseline)
    naive_coords_int = (def_coords_norm * 63).clamp(0, 63).int().cpu().numpy()

    # Corrected deform: network predicts updated features
    print("Running feature correction network...")
    with torch.no_grad():
        corrected_feats = model(coords_norm, feats_t, def_coords_norm)

    corrected_coords_int = (def_coords_norm * 63).clamp(0, 63).int().cpu().numpy()
    corrected_feats_np   = corrected_feats.cpu().numpy()

    # Build SparseTensors
    slat_orig      = make_slat_tensor(orig_coords.astype(np.int32),    orig_feats)
    slat_naive     = make_slat_tensor(naive_coords_int,                 orig_feats)
    slat_corrected = make_slat_tensor(corrected_coords_int,             corrected_feats_np)

    # Render comparison
    print("Rendering...")
    render_comparison(pipeline, slat_orig, slat_naive, slat_corrected, args.out)
    print("Done.")


if __name__ == "__main__":
    main()


# # Use any object from HSSD as test input
# python inference.py \
#     --npz /mnt/val_data/datasets/HSSD_FROM_55/HSSD/latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/001065d01ce8bf908b78a03f32ae36d4e866faee1252ced8488b1225666f7d52.npz \
#     --deform squish \
#     --strength 0.6 \
#     --out results/sneaker_squish