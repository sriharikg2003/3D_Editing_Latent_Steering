#!/usr/bin/env python3
"""
New pipeline: image + text prompt → 3D edit (no target image required).

Changes vs the original inference2.py
--------------------------------------
1. Source image  →  3D generation (run_custom)       [same]
2. Front-view render                                  [same]
3. Text prompt   →  3D voxel mask (create_mask_3d)   [NEW – replaces image editing step]
4. tar_cond is built from TAR_IMAGE_PATH below        [user sets this variable manually]
5. Editing uses the text mask; geometry inside the
   mask is regenerated, outside is preserved          [NEW – replaces filter_edit_regions]
6. Alpha-blending loop kept                           [same]
"""

import os
import bpy
import copy
import torch
import numpy as np
from PIL import Image
from typing import List, Optional

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

torch.set_grad_enabled(False)

from inference.image_processing import bg_to_white, resize_to_512
from inference.rendering        import render_front_view
from inference.model_utils      import (
    load_sparse_structure_encoder,
    inject_methods,
    VoxelProcessor,
    get_dense_cube_fast,
)
from inference.create_mask      import create_mask_3d

import open3d as o3d

# ============================================================================
# USER CONFIGURATION  ← edit these variables before running
# ============================================================================

SRC_INPUT_IMAGE_PATH = "images/lamp.jpeg"  # image to generate 3D from
# TAR_IMAGE_PATH       = "images/image_sofa1.png" # image used as target condition
TEXT_PROMPT          = " modify only the bulb enclosure. Keep the base, pole, arm curvature, proportions, materials, and overall design identical. Change the bulb housing from a short rounded dome into a longer, vertically elongated downward side and dome shape. mask should extend further downward,Ensure smooth continuity with the existing fixture and preserve the original style and scale. Do not mask any other part of the lamp. " # region to edit
OUTPUT_DIR           = "outputs/new_pipeline"

EDITING_SEED   = 1
ST_STEP        = 12          # aggressiveness of sparse-structure editing
DILATION_VOXELS = 10       # how many voxels to expand the mask outward
FORMATS        = ['mesh', 'gaussian', 'radiance_field']

# ============================================================================
# STEP-0: Load models
# ============================================================================

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline = load_sparse_structure_encoder(pipeline)
pipeline = inject_methods(pipeline)
print("Loading TRELLIS pipeline Done")


# ============================================================================
# run_with_mask — core editing function
# ============================================================================

@torch.no_grad()
def run_with_mask(
    pipeline,
    src_image_path:           str,
    src_ply_path:             str,        # voxels.ply from generation step
    src_voxel_latent_path:    str,        # latent.pt  from generation step
    src_slat,                             # SLAT object from run_custom
    tar_cond:                 dict,       # pre-built by caller via pipeline.get_cond()
    mask_ply_path:            str,        # dilated voxel mask from create_mask_3d
    cube_mask_ply_path:       str,        # dense bounding-box mask from create_mask_3d
    num_samples:              int            = 1,
    seed:                     int            = 42,
    sparse_structure_sampler_params: Optional[dict] = None,
    slat_sampler_params:      Optional[dict] = None,
    formats:                  List[str]      = FORMATS,
    preprocess_image:         bool = True,
    output_path:              str  = "",
    st_step:                  int  = 12,
) -> dict:
    """
    Alpha-loop editing using a pre-computed text-guided voxel mask.

    Geometry inside the mask  →  regenerated from tar_cond
    Geometry outside the mask →  preserved from source
    SLAT inside cube mask     →  new appearance from tar_cond
    SLAT outside cube mask    →  preserved from source
    """

    sparse_structure_sampler_params = sparse_structure_sampler_params or {}
    slat_sampler_params             = slat_sampler_params or {}

    src_img = Image.open(src_image_path)
    thick_img = Image.open("/data/home/divya1/projects/assign/Nano3D/images/image_sofa1.png")
    thin_image = Image.open("/data/home/divya1/projects/assign/Nano3D/images/image_sofa.png")
    if preprocess_image:
        src_img = pipeline.preprocess_image(src_img)
        thick_img = pipeline.preprocess_image(thick_img)
        thin_image = pipeline.preprocess_image(thin_image)

    src_cond = pipeline.get_cond([src_img])

    thick_cond = pipeline.get_cond([thick_img])
    thin_cond = pipeline.get_cond([thin_image])

    tar_cond["cond"] = thick_cond["cond"] - thin_cond["cond"]
    tar_cond["neg_cond"] = torch.zeros_like(tar_cond["cond"])

    src_voxel_latent = torch.load(src_voxel_latent_path, weights_only=False)

    # Load text-guided masks
    proc           = VoxelProcessor(grid_size=64)
    text_mask_grid = proc.ply_to_voxel(mask_ply_path)       # (64,64,64) edit region
    cube_mask_grid = proc.ply_to_voxel(cube_mask_ply_path)  # dense bbox of edit region
    v_src          = proc.ply_to_voxel(src_ply_path)        # source geometry

    alphas     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    all_results = {}

    for alpha in alphas:
        cur_out = os.path.join(output_path, f"alpha_{alpha:.1f}")
        os.makedirs(cur_out, exist_ok=True)
        print(f"\n{'='*40}\nalpha = {alpha:.1f}\n{'='*40}")

        # ── interpolate conditions ────────────────────────────────────────────
        cur_tar_cond = copy.deepcopy(tar_cond)
        cur_tar_cond["cond"] = (
            src_cond["cond"] + alpha * (tar_cond["cond"] )
        )
        src_norm = src_cond["cond"].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cur_norm = cur_tar_cond["cond"].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cur_tar_cond["cond"] = cur_tar_cond["cond"] * (src_norm / cur_norm)

        # ── sample new sparse structure (geometry) ────────────────────────────
        torch.manual_seed(seed)
        cur_sparse_params = {
            "source_voxel_latent": src_voxel_latent,
            "src_cond":            src_cond,
            "tar_cond":            cur_tar_cond,
            "output_path":         cur_out,
            "st_step":             st_step,
            **sparse_structure_sampler_params,
        }
        pipeline.sample_sparse_structure(src_cond, num_samples, cur_sparse_params)

        # ── geometry merge using text mask ────────────────────────────────────
        # New voxels from the sampled sparse structure
        new_ply  = os.path.join(cur_out, "edit_voxel.ply")
        v_new    = proc.ply_to_voxel(new_ply)

        # Inside text mask → take new geometry; outside → keep source
        v_merged = v_src.copy()
        v_merged[text_mask_grid > 0] = v_new[text_mask_grid > 0]

        # Save merged geometry
        proc.voxel_to_ply(v_merged, os.path.join(cur_out, "edit_voxel_merged.ply"))
        proc.voxel_to_ply(text_mask_grid, os.path.join(cur_out, "text_mask.ply"))

        # Build tar_coords from merged voxel grid
        v_merged_t = torch.nonzero(torch.from_numpy(v_merged) > 0).to(torch.int32)
        batch_idx  = torch.zeros((v_merged_t.shape[0], 1), dtype=torch.int32)
        tar_coords = torch.cat([batch_idx, v_merged_t], dim=1).cuda()

        # ── sample SLAT (appearance) with tar_cond ────────────────────────────
        tar_slat = pipeline.sample_slat(cur_tar_cond, tar_coords, slat_sampler_params)

        # ── SLAT merge: outside cube mask → restore source features ──────────
        src_np  = src_slat.coords.cpu().numpy()
        tar_np  = tar_slat.coords.cpu().numpy()

        # Build the cube-mask set (inside = edit region, outside = preserved)
        cube_mask_coords = torch.nonzero(torch.from_numpy(cube_mask_grid) > 0).to(torch.int32)
        batch_idx_m      = torch.zeros((cube_mask_coords.shape[0], 1), dtype=torch.int32)
        cube_mask_coords = torch.cat([batch_idx_m, cube_mask_coords], dim=1)
        mask_set         = set(tuple(p) for p in cube_mask_coords.cpu().numpy())

        src_lookup  = {tuple(p): i for i, p in enumerate(src_np)}
        colors      = np.zeros((len(tar_np), 3))
        merge_num   = 0

        for i_tar, pos in enumerate(tar_np):
            pos_t = tuple(pos)
            if pos_t in mask_set:
                # Inside edit region → keep new generated features
                colors[i_tar] = [1.0, 0.0, 0.0]
            else:
                # Outside edit region → restore from source
                if pos_t in src_lookup:
                    tar_slat.feats[i_tar] = src_slat.feats[src_lookup[pos_t]]
                    colors[i_tar] = [0.0, 1.0, 0.0]
                    merge_num += 1
                else:
                    colors[i_tar] = [0.4, 0.4, 0.4]

        print(f"SLAT merge: restored {merge_num}/{len(tar_np)} features from source")

        # Visualisation point cloud (green=preserved, red=edited, grey=new/unmatched)
        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tar_np[:, -3:])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(cur_out, "slat_merge_viz.ply"), pcd)

        all_results[alpha] = pipeline.decode_slat(tar_slat, formats)

    return all_results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_dir = os.path.join(OUTPUT_DIR, "image")
    os.makedirs(image_dir, exist_ok=True)

    # ── STEP 1: Generate 3D from input image ─────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: 3D generation from input image")
    print("="*60)

    gen_result = pipeline.run_custom(
        SRC_INPUT_IMAGE_PATH,
        seed        = EDITING_SEED,
        output_path = OUTPUT_DIR,
    )
    src_slat = gen_result["src_slat"]

    with torch.enable_grad():
        src_glb = postprocessing_utils.to_glb(
            gen_result["src_mesh"]["gaussian"][0],
            gen_result["src_mesh"]["mesh"][0],
            simplify     = 0.95,
            texture_size = 1024,
        )
    src_mesh_path = os.path.join(OUTPUT_DIR, "src_mesh.glb")
    src_glb.export(src_mesh_path)
    print(f"Source mesh saved: {src_mesh_path}")

    # ── STEP 2: Render front view of generated mesh ───────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Front-view render")
    print("="*60)

    render_front_view(
        file_path   = src_mesh_path,
        output_dir  = image_dir,
        output_name = "front.png",
    )
    src_image_path = bg_to_white(os.path.join(image_dir, "front.png"))
    print(f"Front view: {src_image_path}")

    # ── STEP 3: Create 3D mask from text prompt ───────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Text-guided 3D mask creation")
    print("="*60)

    mask_dir    = os.path.join(OUTPUT_DIR, "mask")
    mask_result = create_mask_3d(
        render_dir       = image_dir,
        text_prompt      = TEXT_PROMPT,
        output_dir       = mask_dir,
        front_image_name = os.path.basename(src_image_path),
        image_path       = SRC_INPUT_IMAGE_PATH,
        grid_size        = 64,
        dilation_voxels  = DILATION_VOXELS,
        dilation_pixels  = 4,
        voxel_ply_path   = os.path.join(OUTPUT_DIR, "voxels.ply"),
    )
    print(f"Mask voxels: {mask_result['n_masked']}")

    # Compute solid bounding-box mask for SLAT merge
    _proc = VoxelProcessor(grid_size=64)
    _mask_grid = _proc.ply_to_voxel(mask_result["mask_ply"])
    _cube_grid = get_dense_cube_fast(_mask_grid)
    _cube_mask_ply = os.path.join(mask_dir, "cube_mask.ply")
    _proc.voxel_to_ply(_cube_grid, _cube_mask_ply)
    mask_result["cube_mask_ply"] = _cube_mask_ply
    print(f"Cube mask voxels: {int(_cube_grid.sum())}")

    # ── STEP 4: Build tar_cond from user-provided target image ────────────────
    print("\n" + "="*60)
    print("STEP 4: Build tar_cond from target image")
    print("="*60)


    # thick_image = Image.open("/data/home/divya1/projects/assign/Nano3D/images/image_sofa1.png")
    # thin_image = Image.open("/data/home/divya1/projects/assign/Nano3D/images/image_sofa.png")
  
    # thick_img = pipeline.preprocess_image(thick_image)
    # thin_image = pipeline.preprocess_image(thin_image)

    # tar_cond1 = pipeline.get_cond([thick_image])
    # tar_cond2 = pipeline.get_cond([thin_image])
    # tar_cond = {}
    # tar_cond["cond"] = tar_cond1["cond"] - tar_cond2["cond"]
    # tar_cond["neg_cond"] = torch.zeros_like(tar_cond["cond"])
 

    # ── STEP 5: Editing with text-guided mask ─────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Nano3D editing with text-guided mask")
    print("="*60)

    outputs = run_with_mask(
        pipeline               = pipeline,
        src_image_path         = src_image_path,
        src_ply_path           = os.path.join(OUTPUT_DIR, "voxels.ply"),
        src_voxel_latent_path  = os.path.join(OUTPUT_DIR, "latent.pt"),
        src_slat               = src_slat,
       
        mask_ply_path          = mask_result["mask_ply"],
        cube_mask_ply_path     = mask_result["cube_mask_ply"],
        seed                   = EDITING_SEED,
        output_path            = OUTPUT_DIR,
        st_step                = ST_STEP,
        sparse_structure_sampler_params = {"cfg_strength": 7.5},  # ← add this
        slat_sampler_params             = {"cfg_strength": 7.5},  # ← add this
    )

    # ── STEP 6: Export GLB for each alpha ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: Export results")
    print("="*60)

    for alpha, alpha_outputs in outputs.items():
        with torch.enable_grad():
            tar_glb = postprocessing_utils.to_glb(
                alpha_outputs["gaussian"][0],
                alpha_outputs["mesh"][0],
                simplify     = 0.95,
                texture_size = 1024,
            )
        out_path = os.path.join(OUTPUT_DIR, f"alpha_{alpha:.1f}", "edit_mesh.glb")
        tar_glb.export(out_path)
        print(f"Saved: {out_path}")

    print("\nPipeline complete.")
