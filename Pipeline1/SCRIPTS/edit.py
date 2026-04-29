import os
os.environ['SPCONV_ALGO'] = 'native'

import torch
import numpy as np
import open3d as o3d

from trellis.pipelines.trellis_edit_pipeline import TrellisTextTo3DEditingPipeline

MODEL_PATH = "/datafrom_146/srihari/MODELS/TRELLIS-text-xlarge"
OUTPUT_DIR = "/datafrom_146/srihari/TRELLIS_EDIT_OUTPUTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── prompts ───────────────────────────────────────────────────────────────────
# GEN_PROMPT   : what to generate
# EDIT_PROMPT  : what the object should look like AFTER editing
#                (describe the result, not the operation)
#                e.g. "remove the stick" → "A wonderwoman standing" (no stick)
GEN_PROMPT  = "A wonderwoman standing with red stick"
EDIT_PROMPT = "A wonderwoman standing"   # describes result: no stick

# ── edit region ───────────────────────────────────────────────────────────────
# AABB in voxel-grid coords [0, 63].
# We need to find where the stick is. The voxel grid maps the object to a
# normalised cube. Print voxel stats below and adjust these bounds.
# Start broad (right half of object) and narrow down after inspecting output.
EDIT_AABB_MIN = (32, 0,  0)
EDIT_AABB_MAX = (63, 63, 63)

# ── edit mode ─────────────────────────────────────────────────────────────────
# "appearance" : SLatFlow only — changes texture/style, keeps geometry
# "geometry"   : SSFlow + SLatFlow — changes shape too (use for add/remove)
EDIT_MODE    = "geometry"   # removal needs geometry edit

SEED         = 1
STEPS        = 50
CFG_STRENGTH = 7.5
NUM_RESAMPLE = 3
T_START      = 1.0
FORMATS      = ["mesh", "gaussian"]


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(outputs, tag):
    if "mesh" in outputs:
        r      = outputs["mesh"][0]
        verts  = r.vertices.cpu().numpy()
        faces  = r.faces.long().cpu().numpy()
        mesh   = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if r.vertex_attrs is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                r.vertex_attrs[:, :3].cpu().numpy().clip(0, 1)
            )
        mesh.compute_vertex_normals()
        path = os.path.join(OUTPUT_DIR, f"{tag}_mesh.ply")
        o3d.io.write_triangle_mesh(path, mesh)
        print(f"  saved -> {path}")

    if "gaussian" in outputs:
        path = os.path.join(OUTPUT_DIR, f"{tag}_gaussian.ply")
        outputs["gaussian"][0].save_ply(path)
        print(f"  saved -> {path}")


def print_voxel_stats(coords):
    """Print voxel distribution so you can set a sensible AABB."""
    xyz = coords[:, 1:].float()
    print(f"  voxel count : {coords.shape[0]}")
    print(f"  x range     : [{xyz[:,0].min():.0f}, {xyz[:,0].max():.0f}]")
    print(f"  y range     : [{xyz[:,1].min():.0f}, {xyz[:,1].max():.0f}]")
    print(f"  z range     : [{xyz[:,2].min():.0f}, {xyz[:,2].max():.0f}]")
    print(f"  x median    : {xyz[:,0].median():.0f}")
    print(f"  y median    : {xyz[:,1].median():.0f}")
    print(f"  z median    : {xyz[:,2].median():.0f}")


def save_masked_voxels(coords, mask, tag):
    """Save masked voxels as a point cloud PLY so you can visually verify the mask."""
    masked_xyz = coords[mask, 1:].float().cpu().numpy() / 63.0  # normalise to [0,1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(masked_xyz)
    colors = np.tile([1.0, 0.0, 0.0], (masked_xyz.shape[0], 1))  # red
    pcd.colors = o3d.utility.Vector3dVector(colors)
    path = os.path.join(OUTPUT_DIR, f"{tag}_masked_voxels.ply")
    o3d.io.write_point_cloud(path, pcd)
    print(f"  saved masked voxel cloud -> {path}")

    unmasked_xyz = coords[~mask, 1:].float().cpu().numpy() / 63.0
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(unmasked_xyz)
    colors2 = np.tile([0.5, 0.5, 0.5], (unmasked_xyz.shape[0], 1))  # grey
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    path2 = os.path.join(OUTPUT_DIR, f"{tag}_unmasked_voxels.ply")
    o3d.io.write_point_cloud(path2, pcd2)
    print(f"  saved unmasked voxel cloud -> {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print("Loading pipeline ...")
pipeline = TrellisTextTo3DEditingPipeline.from_pretrained(MODEL_PATH)
pipeline.cuda()

# ── 1. Generate original ──────────────────────────────────────────────────────
print("\n[1/4] Generating original asset ...")
orig_outputs = pipeline.run(GEN_PROMPT, num_samples=1, seed=SEED, formats=FORMATS)
save_outputs(orig_outputs, "original")

# ── 2. Extract exact slat_known with same seed ────────────────────────────────
print("\n[2/4] Extracting slat_known ...")
cond = pipeline.get_cond([GEN_PROMPT])
torch.manual_seed(SEED)
coords = pipeline.sample_sparse_structure(
    cond, num_samples=1,
    sampler_params=pipeline.sparse_structure_sampler_params,
)
torch.manual_seed(SEED)
slat_known = pipeline.sample_slat(
    cond, coords,
    sampler_params=pipeline.slat_sampler_params,
)
print_voxel_stats(coords)

# ── 3. Build mask ─────────────────────────────────────────────────────────────
# INSPECT: open original_masked_voxels.ply and original_unmasked_voxels.ply
# in MeshLab/CloudCompare to verify the mask covers the right region.
# Then adjust EDIT_AABB_MIN / EDIT_AABB_MAX above and re-run.
print("\n[3/4] Building edit mask ...")
mask = TrellisTextTo3DEditingPipeline.mask_from_aabb(
    coords, aabb_min=EDIT_AABB_MIN, aabb_max=EDIT_AABB_MAX,
)
print(f"  masked   : {mask.sum().item()} / {coords.shape[0]} voxels  "
      f"({100*mask.float().mean():.1f}%)")
assert mask.sum()    > 0,              "Mask is empty — widen EDIT_AABB"
assert (~mask).sum() > 50,             "Too few anchor voxels — narrow EDIT_AABB"
assert mask.float().mean() < 0.9,     "Mask covers >90% of object — too large"

save_masked_voxels(coords, mask, "debug")
print("  Open debug_masked_voxels.ply (red) in MeshLab to verify mask covers the stick.")

# ── 4. Edit ───────────────────────────────────────────────────────────────────
# For REMOVAL: use run_edit_geometry — Stage 1 (SSFlow) regenerates the voxel
#   occupancy conditioned on EDIT_PROMPT (no stick), Stage 2 (SLatFlow)
#   regenerates features on the new occupancy.
# For RETEXTURE: use run_edit_appearance — structure stays, only colors change.
print(f"\n[4/4] Running {EDIT_MODE} edit ...")
o3d_mesh = orig_outputs["mesh"][0]
o3d_mesh_converted = o3d.geometry.TriangleMesh()
o3d_mesh_converted.vertices  = o3d.utility.Vector3dVector(o3d_mesh.vertices.cpu().numpy())
o3d_mesh_converted.triangles = o3d.utility.Vector3iVector(o3d_mesh.faces.long().cpu().numpy())

if EDIT_MODE == "geometry":
    edit_outputs = pipeline.run_edit_geometry(
        mesh         = o3d_mesh_converted,
        slat_known   = slat_known,
        mask         = mask,
        prompt       = EDIT_PROMPT,
        steps        = STEPS,
        cfg_strength = CFG_STRENGTH,
        num_resample = NUM_RESAMPLE,
        t_start      = T_START,
        formats      = FORMATS,
        verbose      = True,
    )
else:
    edit_outputs = pipeline.run_edit_appearance(
        slat_known   = slat_known,
        mask         = mask,
        prompt       = EDIT_PROMPT,
        steps        = STEPS,
        cfg_strength = CFG_STRENGTH,
        num_resample = NUM_RESAMPLE,
        t_start      = T_START,
        formats      = FORMATS,
        verbose      = True,
    )

save_outputs(edit_outputs, "edited")

print(f"\nDone. Outputs in {OUTPUT_DIR}/")
print("  original_mesh.ply  original_gaussian.ply")
print("  edited_mesh.ply    edited_gaussian.ply")
print("  debug_masked_voxels.ply  debug_unmasked_voxels.ply")