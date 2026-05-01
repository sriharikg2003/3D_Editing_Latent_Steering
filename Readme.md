# 3D Editing via Latent Steering

**Course:** DLCV  
**Team:** Divya, Srihari K G  
**Built on:** [TRELLIS](https://github.com/microsoft/TRELLIS) (image-to-3D) and [Nano3D](https://arxiv.org/abs/2510.15019)

---

## Project Overview

This project explores **latent-space editing of 3D objects** without retraining diffusion models. The core insight is that the DINOv2 conditioning embeddings and structured latent (SLAT) features produced by TRELLIS live in smooth, disentangled spaces — geometric and appearance attributes can be edited by computing and applying direction vectors directly in those spaces.

Three complementary pipelines are provided, each targeting a different editing scenario:

| Pipeline | Approach | Edit signal | Alpha range |
|----------|----------|-------------|-------------|
| [Pipeline1](#pipeline1--latent-direction-steering) | Directional steering in conditioning / SLAT space | Reference image pair | 0.0 → 1.0 (11 steps) |
| [Pipeline2](#pipeline2--flow-based-latent-interpolation) | FlowEdit velocity blending between two objects | Source + target image | 0.0 → 1.0 (N steps) |
| [Nano3D (mask)](#nano3d-mask--text-guided-3d-editing) | Text-guided voxel masking + proxy direction transfer | Text prompt only | 0.0 → 1.5 (8 steps) |

---

## Repository Layout

```
3D_Editing_Latent_Steering/
├── Pipeline1/                  Latent direction steering (dataset-scale experiments)
├── Pipeline2/                  Flow-based interpolation between two 3D objects
├── Nano_3d(mask)/              Text-guided masked 3D editing (no reference image)
└── Readme.md                   ← this file
```

---

## Pipeline1 — Latent Direction Steering

**Location:** [`Pipeline1/`](Pipeline1/)  
**Entry point:** `Pipeline1/find_direction_interpolate.py`

Computes a **semantic direction vector** from a reference image pair (e.g. thin → thick) and transfers it onto any target object, without the pair needing to show the same object as the target.

### Steering Methods

| Method | Edit space | Description |
|--------|-----------|-------------|
| `run_sparse_interp_with_direction` | DINOv2 conditioning | Transfers direction from reference pair onto a target; geometry-only edit |
| `run_edit_sparse` | DINOv2 conditioning | SLERP interpolation between two images' conditioning vectors |
| `run_directional_edit` | SLAT feature space | Direction from plus/minus pair applied to base object's SLAT features |
| `run_load_dino_vec` | DINOv2 (spatial crop) | Applies a pre-computed difference vector to a centre-crop of patch tokens |
| `run_move_in_latent` | SLAT features | Directly perturbs SLAT features with a noise vector |

A **RePaint-based masked editing pipeline** (`TrellisTextTo3DEditingPipeline`) is also included, supporting text-conditioned region edits on existing 3D assets via an axis-aligned bounding box mask.

### Main Experiment

`find_direction_interpolate.py` runs direction transfer across **15 HSSD dataset assets** for **6 semantic pairs** at α ∈ {0.0, 0.1, …, 1.0}, rendering Gaussian splat videos for each:

| Pair name | Semantic change |
|-----------|----------------|
| `slim_to_thick` | Slim → thick object |
| `squeeze_down` | Vertical scale compression |
| `cyl_to_cube` | Cylinder → cuboid |
| `cyl_to_timer` | Cylinder → timer shape |
| `thin_to_thick_leg_chair` | Thin-legged → thick-legged chair |
| `carve_out_center` | Solid → centre-carved |

Both forward and reverse directions are run; output videos and grid images are saved under `SUBMISSION_OUTPUTS/<pair>/forward|reverse/<asset_id>/`.

### Unsupervised Direction Discovery

`SCRIPTS/train_discovery.py` learns N orthogonal edit directions in SLAT space using a classification objective (InterfaceGAN-style). `SCRIPTS/slider_train.py` discovers a single supervised direction via a binary attribute classifier.

### Quick Start

```bash
export SPCONV_ALGO=native

# Direction-transfer experiment (set PAIR_IDX in script to 0–5)
python Pipeline1/find_direction_interpolate.py

# Masked text-guided editing
python Pipeline1/SCRIPTS/edit.py

# Unsupervised direction discovery
python Pipeline1/SCRIPTS/train_discovery.py
```

> **Note:** Model weights are loaded from a local path (`TRELLIS_MODELS/TRELLIS-image-large`). Update this path before running. Input image pairs must be placed in `Pipeline1/INPUT_IMAGE_PAIRS/`.

---

## Pipeline2 — Flow-Based Latent Interpolation

**Location:** [`Pipeline2/`](Pipeline2/)  
**Entry point:** `Pipeline2/inference_interpolation.py`

Generates a smooth sequence of 3D meshes morphing from a source object to a target object by blending FlowEdit denoising velocities and SLAT appearance features.

### How It Works

1. **Source reconstruction** — TRELLIS encodes the source image → sparse voxel latent + SLAT tensor.
2. **FlowEdit velocity delta** — At each α, source and target velocities are blended: `ΔV = α · (V_tar − V_src)`, steering the sparse structure toward the target geometry.
3. **SLAT interpolation** — Appearance is blended: `(1−α)·slat_src + α·slat_tar`.
4. **Mesh decoding** — Edited latent is decoded to `.glb` and optionally rendered.

### Quick Start

```bash
# Run interpolation (N=6 steps from source to target)
CUDA_VISIBLE_DEVICES=0 python Pipeline2/inference_interpolation.py \
    --src_image Pipeline2/thin.png \
    --tar_image Pipeline2/thick.png \
    --output_dir ./output_interp \
    --editing_mode add \
    --n_steps 6

# Multi-view rendering of all alpha steps (requires Blender)
python Pipeline2/render_interpolation.py \
    --interp_dir ./output_interp \
    --engine BLENDER_EEVEE \
    --resolution 512
```

### Editing Modes

| Mode | Behaviour |
|------|----------|
| `add` | Merges source appearance into regions untouched by the edit |
| `remove` | Inverts the edit direction |
| `replace` | Restricts edits to a masked bounding box region only |

### Sample Image Pairs (included)

| Pair | Source | Target |
|------|--------|--------|
| Sword thickness | `thin.png` | `thick.png` |
| Cylinder thickness | `thin_cyl.png` | `thick_cyl.png` |
| Carrot | `carrot_thin.png` | `carrot_thick.png` |
| Yellow object size | `y_s.png` | `y_b.png` |
| Hat size | `small_hat.png` | `big_hat.png` |

---

## Nano3D (mask) — Text-Guided 3D Editing

**Location:** [`Nano_3d(mask)/`](Nano_3d(mask)/)  
**Entry point:** `Nano_3d(mask)/new_pipeline.py`

Extends Nano3D to enable editing from a **text prompt alone** — no pre-edited reference image required. A 3D voxel mask is automatically generated from the text description, preserving everything outside the target region precisely.

### How It Works

```
Input: source image  +  text prompt (e.g. "the bulb")
Output: edited 3D mesh at α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5}
```

**Step 1** — Generate 3D from source image via TRELLIS (`run_custom`).  
**Step 2** — Render front view with Blender; save camera metadata for ray casting.  
**Step 3** — Build a 3D voxel mask from the text prompt:

```
Text prompt
    → Grounding DINO  (smallest valid bounding box on source image)
    → SAM             (2D segmentation mask on rendered front view)
    → Ray casting     (first-hit voxel per masked pixel)
    → 3D dilation     (dilated voxel mask + dense bounding-box cube mask)
```

**Step 4** — Compute edit direction from a proxy image pair:

```
direction = encode(thick_proxy) − encode(thin_proxy)
```

**Step 5** — For each α:
1. Steer conditioning: `cur_cond = src_cond + α × direction` (norm-preserved)
2. Run FlowEdit → new geometry
3. Voxel merge: inside text mask → new geometry; outside → source geometry
4. Sample new SLAT appearance; SLAT merge: inside cube mask → new, outside → source
5. Decode and export `edit_mesh.glb`

The α sweep up to 1.5 (super-linear amplification) lets you explore edits beyond the proxy pair's natural magnitude.

### Quick Start

```bash
# 1. Edit SRC_INPUT_IMAGE_PATH, TEXT_PROMPT, OUTPUT_DIR at the top of new_pipeline.py
# 2. Update the proxy pair image paths inside run_with_mask (~line 102)
python Nano_3d(mask)/new_pipeline.py

# Sanity-check text-to-2D mask before running the full pipeline
python Nano_3d(mask)/new_mask.py
```

### Comparison with Original Nano3D

| | Original Nano3D | This Work |
|---|---|---|
| Edit signal | Pre-edited image (Qwen-Image or manual) | Text prompt + proxy direction pair |
| Region mask | None — global FlowEdit | Text-guided 3D voxel mask |
| Geometry merge | XOR + connected-component filtering | Voxel mask gates merge |
| SLAT merge | Source overwrites target at all shared coords | Cube mask: edit inside, preserve outside |
| Conditioning | Single target image interpolated to source | `src + α × (thick − thin)` direction (norm-preserved) |
| Alpha sweep | 6 outputs at α = 0.0 … 1.0 | 8 outputs at α = 0.0 … 1.5 |

---

## Dependencies

| Category | Packages |
|----------|----------|
| Deep learning | `torch==2.4.0+cu118`, `diffusers==0.34.0`, `transformers`, `accelerate` |
| Sparse 3D | `spconv-cu118`, `kaolin` |
| 3D processing | `open3d`, `trimesh`, `pymeshfix`, `pyvista`, `plyfile`, `xatlas`, `pysdf` |
| Vision | `rembg`, `opencv-python`, `dinov2` (via `torch.hub`) |
| Rendering | `bpy==4.0.0` (Blender), `imageio[ffmpeg]` |
| Masking (Nano3D) | `IDEA-Research/grounding-dino-base`, `facebook/sam-vit-base`, `CIDAS/clipseg-rd64-refined` |

```bash
# Required environment variable for all pipelines
export SPCONV_ALGO=native

# Build the voxel-sequence CUDA extension (Pipeline2 / Nano3D)
cd <pipeline>/extensions/vox2seq && python setup.py install
```

**Model weights:**
- Pipeline1 loads weights from a local path (`TRELLIS_MODELS/TRELLIS-image-large`) — update path in scripts.
- Pipeline2 and Nano3D download `microsoft/TRELLIS-image-large` automatically from HuggingFace on first run.

**Hardware:** GPU with ≥ 24 GB VRAM recommended (tested on A100/H100).

---

## Built On

- [TRELLIS](https://github.com/microsoft/TRELLIS) — sparse 3D generation backbone (sparse-structure flow + SLAT decoder)
- [Nano3D](https://arxiv.org/abs/2510.15019) — FlowEdit integration and Voxel/SLAT-Merge strategy
- [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) — open-vocabulary object detection
- [SAM](https://huggingface.co/facebook/sam-vit-base) — Segment Anything Model
- [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) — text-conditioned 2D segmentation
