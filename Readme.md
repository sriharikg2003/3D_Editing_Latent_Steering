# 3D Editing via Latent Steering

**Course:** DLCV  
**Team:** Divya, Srihari K G  
**Built on:** [TRELLIS](https://github.com/microsoft/TRELLIS) (image/text-to-3D) and [Nano3D](https://arxiv.org/abs/2510.15019)

---

## Project Overview

This project explores **latent-space editing of 3D objects** without retraining diffusion models. The core insight is that the DINOv2 conditioning embeddings and structured latent (SLAT) features produced by TRELLIS live in smooth, disentangled spaces — geometric and appearance attributes can be edited by computing and applying direction vectors directly in those spaces.

Three complementary pipelines are provided:

| Pipeline | Approach | Edit signal |
|----------|----------|-------------|
| [Pipeline1](#pipeline1--latent-direction-steering) | Directional steering in conditioning / SLAT space | Reference image pair |
| [Pipeline2](#pipeline2--flow-based-latent-interpolation) | Flow-based velocity interpolation between two objects | Source + target image |
| [Nano3D (mask)](#nano3d-mask--text-guided-3d-editing) | Text-guided voxel masking + direction transfer | Text prompt only |

---

## Pipeline1 — Latent Direction Steering

**Location:** [`Pipeline1/`](Pipeline1/)

Extends TRELLIS image-to-3D with several latent-space editing methods. A **direction vector** is computed from a reference image pair (e.g. thin→thick) and transferred onto any target object.

### Steering Methods

| Method | Space | Description |
|--------|-------|-------------|
| `run_sparse_interp_with_direction` | DINOv2 conditioning | Transfers a semantic change direction from a reference pair onto a target image |
| `run_edit_sparse` | DINOv2 conditioning | SLERP interpolation between two images' conditioning vectors |
| `run_directional_edit` | SLAT feature space | Computes direction from plus/minus image pair, applies to base object |
| `run_load_dino_vec` | DINOv2 (spatial subset) | Applies a pre-computed difference vector to a spatial crop of patch tokens |
| `run_move_in_latent` | SLAT features | Directly perturbs SLAT features with a noise vector |

Additionally, a **RePaint-based masked editing pipeline** (`TrellisTextTo3DEditingPipeline`) supports text-conditioned region edits on existing 3D assets using an axis-aligned bounding box mask.

### Main Experiment

`find_direction_interpolate.py` runs direction transfer across 15 HSSD dataset assets for 6 semantic pairs (slim→thick, cylinder→cuboid, carve-out center, etc.) at α ∈ {0.0, 0.1, …, 1.0}, rendering Gaussian splat videos for each.

### Quick Start

```bash
export SPCONV_ALGO=native

# Direction-transfer experiment (set PAIR_IDX in script)
python Pipeline1/find_direction_interpolate.py

# Masked text-guided editing
python Pipeline1/SCRIPTS/edit.py

# Unsupervised direction discovery
python Pipeline1/SCRIPTS/train_discovery.py
```

---

## Pipeline2 — Flow-Based Latent Interpolation

**Location:** [`Pipeline2/`](Pipeline2/)

Enables smooth morphing between two 3D objects by interpolating in TRELLIS latent space. Given source and target images, it generates a sequence of intermediate meshes using **FlowEdit velocity blending**.

### How It Works

1. **Source reconstruction** — TRELLIS encodes the source image into a sparse voxel latent and a SLAT tensor.
2. **FlowEdit velocity delta** — At each step α, denoising velocities for source and target are blended: `ΔV = α · (V_tar − V_src)`.
3. **SLAT interpolation** — Appearance is blended linearly: `(1−α)·slat_src + α·slat_tar`.
4. **Mesh decoding** — The edited latent is decoded to a `.glb` mesh.

### Quick Start

```bash
# Run interpolation
CUDA_VISIBLE_DEVICES=0 python Pipeline2/inference_interpolation.py \
    --src_image thin.png \
    --tar_image thick.png \
    --output_dir ./output_interp \
    --editing_mode add \
    --n_steps 6

# Multi-view rendering (optional, requires Blender)
python Pipeline2/render_interpolation.py \
    --interp_dir ./output_interp \
    --engine BLENDER_EEVEE \
    --resolution 512
```

### Editing Modes

| Mode | Behavior |
|------|----------|
| `add` | Merges source appearance into regions untouched by the edit |
| `remove` | Inverts the edit direction (removes a feature) |
| `replace` | Restricts edits to a masked region only |

---

## Nano3D (mask) — Text-Guided 3D Editing

**Location:** [`Nano_3d(mask)/`](Nano_3d(mask)/)

Extends Nano3D to enable 3D editing from a **text prompt alone** — no pre-edited reference image required. A 3D voxel mask is automatically generated from the text description to preserve everything outside the target region.

### How It Works

```
Input: source image  +  text prompt (e.g. "the bulb")
Output: edited 3D mesh at α = 0.0 → 1.0
```

**Step 1** — Generate 3D from source image via TRELLIS.  
**Step 2** — Render front view and save camera metadata.  
**Step 3** — Create 3D voxel mask via text prompt:

```
Text prompt → Grounding DINO (bounding box) → SAM (2D mask)
    → Ray casting (3D surface mask) → 3D dilation (interior voxels)
```

**Step 4** — Compute DINOv2 direction: `direction = encode(tar_image) − encode(src_image)`.  
**Step 5** — For each α: steer conditioning, run FlowEdit, merge voxels/SLAT inside mask, decode mesh.

### Quick Start

```bash
# Edit configuration at top of new_pipeline.py, then run:
python Nano_3d(mask)/new_pipeline.py

# Sanity-check text-to-2D-mask before running the full pipeline:
python Nano_3d(mask)/new_mask.py
```

### Comparison with Original Nano3D

| | Original Nano3D | This Work |
|---|---|---|
| Edit signal | Pre-edited image (Qwen-Image or manual) | Text prompt only |
| Region mask | None — global FlowEdit | Text-guided 3D voxel mask |
| Conditioning | Single target image embedding | `tar − src` direction vector |
| Alpha sweep | Single output | 6 outputs at α = 0.0 … 1.0 |

---

## Dependencies

| Category | Packages |
|----------|----------|
| Deep learning | `torch>=2.0`, `diffusers`, `transformers`, `accelerate` |
| Sparse 3D | `spconv-cu118` or `torchsparse`, `kaolin` |
| 3D processing | `open3d`, `trimesh`, `pymeshfix`, `pyvista` |
| Vision | `dinov2` (via `torch.hub`), `rembg`, `opencv-python` |
| Rendering | `imageio[ffmpeg]`, `bpy==4.0.0` (Blender, for Pipeline2) |
| Masking (Nano3D) | `IDEA-Research/grounding-dino-base`, `facebook/sam-vit-base`, `CIDAS/clipseg-rd64-refined` |

Set the sparse conv backend before running:
```bash
export SPCONV_ALGO=native
```

Model weights for Pipeline1 are loaded from a local path (`TRELLIS_MODELS/TRELLIS-image-large`). Pipeline2 and Nano3D download weights automatically from HuggingFace on first run.
