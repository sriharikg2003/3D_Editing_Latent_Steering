# Pipeline1 — 3D Latent Steering via TRELLIS

**Course:** DLCV  
**Team:** Divya, Srihari K G  
**Built on:** [TRELLIS](https://github.com/microsoft/TRELLIS) (image/text-to-3D) and Nano3D

---

## Overview

Pipeline1 extends the TRELLIS image-to-3D and text-to-3D frameworks with several latent-space editing and steering methods. The core idea is that the DINOv2 conditioning embeddings and the structured latent (SLAT) features produced by TRELLIS live in smooth, disentangled spaces — and geometric/appearance attributes can be edited by computing and applying **direction vectors** directly in those spaces, without retraining the diffusion backbone.

Two high-level editing surfaces are provided:

| Surface | Where it acts | What changes |
|---|---|---|
| **Conditioning (DINOv2) steering** | Image conditioning tokens fed into the sparse-structure flow model (Stage 1) | Object geometry / structure |
| **SLAT feature steering** | Sparse structured-latent features fed into the SLAT decoder (Stage 2) | Geometry + appearance together |

Additionally, a **RePaint-based masked editing pipeline** allows text-conditioned region edits on existing 3D assets.

---

## Repository Layout

```
Pipeline1/
├── find_direction_interpolate.py   # Main experiment: directional steering on a dataset
├── configs/
│   ├── generation/                 # Flow-model configs (SS / SLAT, B/L/XL, image/text)
│   └── vae/                        # VAE decoder configs (mesh, Gaussian, radiance field)
├── trellis/
│   ├── pipelines/
│   │   ├── trellis_image_to_3d.py  # Core image-to-3D pipeline + all steering methods
│   │   ├── trellis_text_to_3d.py   # Text-to-3D pipeline
│   │   ├── trellis_edit_pipeline.py # RePaint masked-edit pipeline
│   │   └── samplers/               # Flow Euler sampler + CFG / guidance-interval mixins
│   ├── models/                     # Sparse-structure flow, SLAT flow, VAE models
│   ├── modules/                    # Sparse conv, attention, transformer blocks
│   ├── representations/            # Gaussian, mesh (FlexiCubes), radiance-field, octree
│   ├── renderers/                  # Gaussian, mesh, and octree renderers
│   ├── trainers/                   # Flow-matching and VAE trainer base classes
│   ├── datasets/                   # Dataset loaders for sparse-structure / SLAT data
│   └── utils/                      # Rendering, post-processing, loss, grad-clip helpers
└── SCRIPTS/                        # Experiment scripts (described below)
```

---

## Core Pipeline: `TrellisImageTo3DPipeline`

Located in [trellis/pipelines/trellis_image_to_3d.py](trellis/pipelines/trellis_image_to_3d.py).

### Standard inference

```python
from trellis.pipelines import TrellisImageTo3DPipeline
from PIL import Image

pipeline = TrellisImageTo3DPipeline.from_pretrained("TRELLIS_MODELS/TRELLIS-image-large")
pipeline.cuda()

outputs = pipeline.run(Image.open("object.png"), seed=42, formats=["gaussian", "mesh"])
```

The two-stage process:
1. **Stage 1 — Sparse Structure (SS):** DINOv2 encodes the image → sparse-structure flow model samples voxel occupancy → decoder extracts 3D coordinates.
2. **Stage 2 — SLAT:** SLAT flow model fills each occupied voxel with features → decoders produce Gaussian splats, mesh, or radiance field.

### Steering methods

#### 1. `run_sparse_interp_with_direction` — Direction transfer

The key method used in `find_direction_interpolate.py`. Given a **reference pair** (`image1`, `image2`) that encodes a semantic change (e.g. thin→thick), it transfers that change direction onto any **target image**:

```
direction = encode(image2) - encode(image1)          # semantic change direction
direction_normalized = direction * ‖actual‖ / ‖direction‖  # preserve target scale
cond_steered = cond_actual + alpha * direction_normalized
cond_steered = cond_steered * ‖actual‖ / ‖cond_steered‖   # renormalize magnitude
```

`alpha ∈ [0, 1]` controls the edit strength. The sparse structure is sampled with the steered conditioning; the SLAT is sampled with the original (unsteered) conditioning to preserve appearance.

#### 2. `run_edit_sparse` — Spherical interpolation (SLERP)

Interpolates the conditioning vectors of two images using SLERP instead of linear interpolation, producing smoother transitions in the conditioning manifold:

```python
outputs = pipeline.run_edit_sparse(image1, image2, alpha=0.5)
```

#### 3. `run_directional_edit` — SLAT-space direction transfer

Computes a direction in **SLAT feature space** (not conditioning space) from a plus/minus image pair and applies it to a base object:

```python
outputs = pipeline.run_directional_edit(
    image_base=base_img,
    image_plus=thick_img,
    image_minus=thin_img,
    alpha=1.5,
)
```

Direction is computed only on voxels present in both plus and minus SLATs (voxel intersection), or falls back to a global mean-shift.

#### 4. `run_load_dino_vec` — Spatially-masked conditioning steering

Applies a pre-computed DINOv2 difference vector to a spatial subset of patch tokens (center crop of the 37×37 patch grid), allowing localized structural edits:

```python
outputs = pipeline.run_load_dino_vec(image, alpha=0.8, crop_size=10)
```

#### 5. `run_move_in_latent` — Explicit SLAT perturbation

Directly adds a noise vector to SLAT features and decodes:

```python
slat, noise = pipeline.sample_slat_and_noise(image)
outputs = pipeline.run_move_in_latent(noise, slat, alpha=2.0)
```

---

## Masked Editing Pipeline: `TrellisTextTo3DEditingPipeline`

Located in [trellis/pipelines/trellis_edit_pipeline.py](trellis/pipelines/trellis_edit_pipeline.py).

Extends the text-to-3D pipeline with **RePaint**-based masked editing. A user-specified axis-aligned bounding box (AABB) in voxel space defines the region to edit; everything outside is preserved as-is.

Two edit modes:

| Mode | API | Stages regenerated |
|---|---|---|
| Appearance only | `run_edit_appearance` | Stage 2 (SLAT) only |
| Geometry + appearance | `run_edit_geometry` | Stage 1 (SS) + Stage 2 (SLAT) |

### Usage example

```python
from trellis.pipelines.trellis_edit_pipeline import TrellisTextTo3DEditingPipeline

pipeline = TrellisTextTo3DEditingPipeline.from_pretrained("TRELLIS-text-xlarge")
pipeline.cuda()

# Generate original
orig = pipeline.run("A wonderwoman standing with red stick", seed=1)

# Build edit mask (voxel AABB)
mask = TrellisTextTo3DEditingPipeline.mask_from_aabb(coords, (32,0,0), (63,63,63))

# Edit: remove the stick
edited = pipeline.run_edit_geometry(
    mesh=orig["mesh"][0],
    slat_known=slat_known,
    mask=mask,
    prompt="A wonderwoman standing",
    steps=50, cfg_strength=7.5, num_resample=3,
)
```

---

## Main Experiment Script: `find_direction_interpolate.py`

Runs the direction-transfer experiment across a dataset of 15 assets from HSSD, for both forward and reverse directions of 6 pre-defined image pairs:

| Pair | Semantic change |
|---|---|
| `slim_to_thick` | Slim → thick object |
| `squeeze_down` | Scale compression |
| `cyl_to_cube` | Cylinder → cuboid |
| `cyl_to_timer` | Cylinder → timer shape |
| `thin_to_thick_leg_chair` | Thin-legged → thick-legged chair |
| `carve_out_center` | Solid → center-carved |

For each asset and each `alpha ∈ {0.0, 0.1, …, 1.0}`, a Gaussian splat video is rendered and saved. A grid image is assembled from the first frames of all alpha-interpolated videos.

Output structure:
```
SUBMISSION_OUTPUTS/
└── <pair_name>/
    ├── forward/
    │   └── <asset_id>/
    │       ├── 0.0000_gs.mp4
    │       ├── 0.1000_gs.mp4
    │       ├── ...
    │       └── grid.png
    └── reverse/
        └── <asset_id>/
            └── ...
```

---

## Latent Direction Discovery Scripts

### `SCRIPTS/train_discovery.py` — Unsupervised direction discovery

Learns `N` orthogonal edit directions in the sparse-structure latent space using a **classification objective**: a `StructureReconstructor` network is trained to identify which direction was applied and predict its magnitude. The direction matrix `A` is updated to maximize identifiability.

Key parameters:
- `NUM_DIRECTIONS = 12` — number of independent edit directions to discover
- `ITERATIONS = 500` — training iterations
- `LAMBDA_REG = 0.25` — magnitude regression loss weight

Outputs: `checkpoints/directions_A_final.pt`, per-direction render videos.

### `SCRIPTS/slider_train.py` — Supervised direction via InterfaceGAN

Trains a binary classifier on 500 sampled latents scored by a continuous attribute (e.g. thickness). The gradient of the classifier w.r.t. the average latent gives a **semantic edit direction**. This is the InterfaceGAN approach applied to TRELLIS SLAT space.

---

## Dependencies

- PyTorch ≥ 2.0
- `spconv` or `torchsparse` (sparse convolution backend — set `SPCONV_ALGO=native`)
- `rembg` (background removal for image preprocessing)
- `dinov2` (via `torch.hub`, `facebookresearch/dinov2`)
- `open3d` (mesh I/O and visualization)
- `imageio`, `opencv-python` (video rendering)
- `easydict`, `tqdm`

---

## Quick Start

```bash
# Set sparse conv backend
export SPCONV_ALGO=native

# Run direction-transfer experiment (edit PAIR_IDX in the script to select pair)
python find_direction_interpolate.py

# Run masked text-guided editing
python SCRIPTS/edit.py

# Run unsupervised direction discovery
python SCRIPTS/train_discovery.py
```

---

## Notes

- Model weights are loaded from a local path (`TRELLIS_MODELS/TRELLIS-image-large`). Update this path in scripts before running.
- `find_direction_interpolate.py` expects input images in `INPUT_IMAGE_PAIRS/` and dataset assets in `assets/250_SAMPLED_FROM_HSSD/`.
- The noise persistence path in `sample_sparse_structure` (`/mnt/data/srihari/...`) is environment-specific and should be updated for your setup.
