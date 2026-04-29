"""
generate_pairs.py
-----------------
Generates SLAT training pairs for deformation feature learning.

For each object image:
  1. Generate base SLAT via TRELLIS image pipeline
  2. Decode base mesh
  3. Apply 5 deformations (squish levels) to the mesh
  4. Re-encode each deformed mesh through TRELLIS (same image condition)
  5. Save (slat_original, slat_deformed) pairs to disk

Output structure:
  PAIRS/
    obj_000/
      slat_orig.pt          # original SLAT coords + feats
      deform_0.8.pt         # squished SLAT coords + feats
      deform_0.6.pt
      deform_0.4.pt
      deform_twist_0.2.pt
      deform_taper_0.8.pt
      render_orig.png       # for visual inspection
"""

import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

import torch
import numpy as np
import open3d as o3d
import trimesh
import imageio
from PIL import Image
from pathlib import Path

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import trellis.modules.sparse as sp

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/mnt/data/srihari/MODELS/TRELLIS-image-large"
IMAGE_DIR = "BOOTSTRAP"   # folder with .png/.jpg images
OUTPUT_DIR   = "PAIRS"
SAMPLER_PARAMS = {"steps": 12, "cfg_strength": 7.5}
SEED         = 42

# Squish levels along vertical axis (axis index 2 in voxel coords)
SQUISH_LEVELS = [0.8, 0.6, 0.4]
# Twist angles in degrees
TWIST_ANGLES  = [15.0]
# Taper scales (top shrinks, bottom stays)
TAPER_SCALES  = [0.8]
# ──────────────────────────────────────────────────────────────────────────────


def load_pipeline():
    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_PATH)
    pipeline.cuda()
    return pipeline


def encode_image_to_slat(pipeline, image: Image.Image):
    """Run full TRELLIS pipeline, return raw SLAT SparseTensor."""
    cond = pipeline.get_cond([image])
    torch.manual_seed(SEED)
    coords = pipeline.sample_sparse_structure(cond, 1, SAMPLER_PARAMS)
    slat   = pipeline.sample_slat(cond, coords, SAMPLER_PARAMS)
    return slat, cond


def slat_to_mesh(pipeline, slat) -> trimesh.Trimesh:
    """Decode SLAT to mesh, return as trimesh object."""
    decoded = pipeline.decode_slat(slat, formats=['mesh'])
    o3d_mesh = decoded['mesh'][0]
    verts  = np.asarray(o3d_mesh.vertices)
    faces  = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices  = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    return o3d_mesh


def encode_mesh_to_slat(pipeline, mesh: trimesh.Trimesh, cond: dict):
    """Re-encode a (deformed) mesh using existing image conditioning."""
    o3d_mesh = trimesh_to_o3d(mesh)
    coords = pipeline.voxelize(o3d_mesh)
    if coords.shape[0] == 0:
        return None
    coords = torch.cat([
        torch.zeros(coords.shape[0], 1).int().cuda(),
        coords
    ], dim=1)
    torch.manual_seed(SEED)
    slat = pipeline.sample_slat(cond, coords, SAMPLER_PARAMS)
    return slat


# ── Deformation functions ─────────────────────────────────────────────────────

def squish_mesh(mesh: trimesh.Trimesh, scale: float) -> trimesh.Trimesh:
    """Compress along Y axis (vertical) around centroid."""
    verts  = mesh.vertices.copy()
    center = verts.mean(0)
    verts[:, 1] = (verts[:, 1] - center[1]) * scale + center[1]
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)


def twist_mesh(mesh: trimesh.Trimesh, max_angle_deg: float) -> trimesh.Trimesh:
    """Twist around Y axis — rotation angle proportional to height."""
    verts  = mesh.vertices.copy()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_range = y_max - y_min + 1e-8
    for i, v in enumerate(verts):
        t     = (v[1] - y_min) / y_range          # 0 at bottom, 1 at top
        angle = np.radians(max_angle_deg * t)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, z  = v[0], v[2]
        verts[i, 0] =  cos_a * x + sin_a * z
        verts[i, 2] = -sin_a * x + cos_a * z
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)


def taper_mesh(mesh: trimesh.Trimesh, top_scale: float) -> trimesh.Trimesh:
    """Taper: shrink XZ at top, keep bottom unchanged."""
    verts  = mesh.vertices.copy()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_range = y_max - y_min + 1e-8
    center_xz = verts[:, [0, 2]].mean(0)
    for i, v in enumerate(verts):
        t     = (v[1] - y_min) / y_range          # 0 at bottom, 1 at top
        scale = 1.0 - (1.0 - top_scale) * t
        verts[i, 0] = center_xz[0] + (v[0] - center_xz[0]) * scale
        verts[i, 2] = center_xz[1] + (v[2] - center_xz[1]) * scale
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_slat(slat, path: str):
    torch.save({'coords': slat.coords.cpu(), 'feats': slat.feats.cpu()}, path)
    print(f"  Saved {path}  coords={slat.coords.shape}  feats={slat.feats.shape}")


def save_render(pipeline, slat, path: str):
    decoded = pipeline.decode_slat(slat, formats=['gaussian'])
    frame   = render_utils.render_video(decoded['gaussian'][0], num_frames=1)['color'][0]
    imageio.imwrite(path, frame)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pipeline = load_pipeline()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect images
    image_paths = sorted([
        p for p in Path(IMAGE_DIR).iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
    ])
    print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

    for obj_idx, img_path in enumerate(image_paths):
        out_dir = f"{OUTPUT_DIR}/obj_{obj_idx:03d}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{obj_idx+1}/{len(image_paths)}] Processing {img_path.name}")

        # Load image
        image = Image.open(img_path / "image.png").convert("RGB")


        # --- Step 1: encode original ---
        print("  Encoding original...")
        slat_orig, cond = encode_image_to_slat(pipeline, image)
        save_slat(slat_orig, f"{out_dir}/slat_orig.pt")
        save_render(pipeline, slat_orig, f"{out_dir}/render_orig.png")

        # --- Step 2: decode to mesh ---
        print("  Decoding to mesh...")
        base_mesh = slat_to_mesh(pipeline, slat_orig)

        # --- Step 3: apply deformations and re-encode ---
        deformations = {}

        for scale in SQUISH_LEVELS:
            deformations[f"squish_{scale}"] = squish_mesh(base_mesh, scale)

        for angle in TWIST_ANGLES:
            deformations[f"twist_{angle}"] = twist_mesh(base_mesh, angle)

        for ts in TAPER_SCALES:
            deformations[f"taper_{ts}"] = taper_mesh(base_mesh, ts)

        for name, deformed_mesh in deformations.items():
            print(f"  Encoding deformation: {name}")
            slat_def = encode_mesh_to_slat(pipeline, deformed_mesh, cond)
            if slat_def is None:
                print(f"  WARNING: voxelization empty for {name}, skipping")
                continue
            save_slat(slat_def, f"{out_dir}/{name}.pt")

        print(f"  Done obj_{obj_idx:03d}")

    print(f"\nAll done. Pairs saved to {OUTPUT_DIR}/")
    print(f"Total objects: {len(image_paths)}")
    print(f"Pairs per object: {len(SQUISH_LEVELS) + len(TWIST_ANGLES) + len(TAPER_SCALES)}")


if __name__ == "__main__":
    main()