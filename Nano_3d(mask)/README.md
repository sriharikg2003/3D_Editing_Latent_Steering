# Text-Guided 3D Editing Without a Reference Image

**DLCV Course Project** — Divya & Srihari K G

This project extends [Nano3D](https://arxiv.org/abs/2510.15019) (built on [TRELLIS](https://github.com/microsoft/TRELLIS)) to enable 3D object editing using only a **text prompt** — no pre-edited reference image required.

---

## Overview

The original Nano3D requires a manually or automatically edited 2D image to guide the 3D edit. We remove this bottleneck by automatically generating a 3D voxel mask from a text description of the region to edit. A latent direction vector then drives the geometry and appearance change, while everything outside the masked region is preserved exactly.

```
Input: source image  +  text prompt ("the bulb")
Output: edited 3D mesh (GLB) at multiple edit strengths (α = 0.0 → 1.0)
```

---

## New Files

| File | Description |
|------|-------------|
| `new_pipeline.py` | End-to-end pipeline: image + text → edited 3D mesh |
| `inference/create_mask.py` | Text-guided 3D voxel mask (GDINO + SAM + ray casting + dilation) |
| `new_mask.py` | Standalone CLIPSeg utility for 2D mask visualisation |

---

## Installation

### Base Environment

Follow the [TRELLIS installation guide](https://github.com/microsoft/TRELLIS) to set up the base environment, then install:

```bash
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install transformers scipy
```

The mask pipeline downloads these models automatically from Hugging Face on first run:
- `IDEA-Research/grounding-dino-base`
- `facebook/sam-vit-base`
- `CIDAS/clipseg-rd64-refined` (used by `new_mask.py` only)

---

## Running the New Pipeline

### `new_pipeline.py` — Image + Text → Edited 3D Mesh

Edit the configuration block at the top of the file:

```python
SRC_INPUT_IMAGE_PATH = "images/lamp.jpeg"   # source image
TEXT_PROMPT          = "the bulb"           # region to edit
OUTPUT_DIR           = "outputs/new_pipeline"

EDITING_SEED         = 1
ST_STEP              = 12    # edit aggressiveness (higher = bigger geometry change)
DILATION_VOXELS      = 10   # how far to expand the 3D mask outward (in voxels)
```

Then run:

```bash
python new_pipeline.py
```

### How it works — step by step

**Step 1 — Generate 3D from source image**

`run_custom` passes the source image through TRELLIS to produce a sparse voxel structure, structured latent (SLAT), and an initial mesh saved as `src_mesh.glb`.

**Step 2 — Render front view**

The generated mesh is rendered from the front. The render and its camera metadata (`front_metadata.json`) are saved to `outputs/image/` and are used by the ray caster in the next step.

**Step 3 — Create 3D voxel mask from text prompt**

`create_mask_3d` builds a 3D mask in four stages:

```
Text prompt
    │
    ▼
Grounding DINO  ──→  Smallest valid bounding box on the source image
    │
    ▼
SAM             ──→  2D binary mask on the rendered front view
    │
    ▼
Ray casting     ──→  3D surface mask (first-hit voxel per masked pixel)
    │
    ▼
3D dilation     ──→  Dilated voxel mask  +  dense bounding-box cube mask
```

- **Grounding DINO** selects the *smallest* box above the score threshold, not the highest-confidence one — this avoids the common failure where the top-scoring box covers the whole object.
- **SAM** segments the rescaled bounding box on the rendered view (not the source image) so the 2D mask is geometrically aligned with the 3D voxel grid.
- **Ray casting** fires one ray per masked pixel from the camera, stopping at the first occupied voxel.
- **3D dilation** expands the surface mask with a spherical kernel to capture interior voxels unreachable from a single viewpoint. A tight axis-aligned bounding box (cube mask) is also computed for use in SLAT merge.

**Step 4 — Build conditioning direction**

Instead of a single target image, a *direction* in DINOv2 embedding space is computed:

```
direction = encode(tar_image) − encode(src_image)
```

This direction captures the semantic change (e.g. thin → thick, short → tall) without requiring an edited version of the specific input object.

**Step 5 — Alpha-loop editing**

For each α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}:

1. Interpolate conditioning: `cur_cond = src_cond + α × direction`
2. Sample new sparse structure (geometry) with FlowEdit using `cur_cond`
3. **Voxel merge** — inside text mask → new geometry; outside → original source geometry
4. Sample new SLAT (appearance) on merged voxel coordinates
5. **SLAT merge** — inside cube mask → new appearance features; outside → original source features
6. Decode and export `edit_mesh.glb`

The α sweep lets you inspect the full range from no change (α = 0) to full edit strength (α = 1).

### Output structure

```
outputs/new_pipeline/
├── src_mesh.glb                   source mesh
├── image/
│   ├── front.png                  rendered front view
│   └── front_metadata.json        camera parameters for ray casting
├── mask/
│   ├── mask_2d.png                2D segmentation mask (debug)
│   ├── mask.ply                   dilated 3D voxel mask
│   └── cube_mask.ply              dense bounding-box cube mask
├── alpha_0.0/
│   ├── edit_mesh.glb
│   ├── edit_voxel.ply             raw geometry from FlowEdit
│   ├── edit_voxel_merged.ply      geometry after voxel merge
│   └── slat_merge_viz.ply         red=edited  green=preserved  grey=new
├── alpha_0.2/
│   └── ...
└── alpha_1.0/
    └── ...
```

---

## Standalone 2D Mask Check (`new_mask.py`)

Before running the full pipeline, you can sanity-check what a text prompt selects in 2D using CLIPSeg:

```bash
python new_mask.py
# Edit IMAGE_PATH and PROMPT at the bottom of the file.
```

Saves three files in the current directory:
- `comparison_plot.png` — source image side-by-side with the mask heatmap
- `heatmap_mask.png` — continuous confidence heatmap
- `binary_mask.png` — thresholded black-and-white mask

---

## `create_mask_3d` API Reference

```python
from inference.create_mask import create_mask_3d

result = create_mask_3d(
    image_path       = "images/lamp.jpeg",    # source image for GDINO detection
    text_prompt      = "the bulb",            # region to edit
    render_dir       = "outputs/image",       # directory with front.png + front_metadata.json
    voxel_ply_path   = "outputs/voxels.ply",  # source voxel grid from TRELLIS generation
    output_dir       = "outputs/mask",
    grid_size        = 64,
    dilation_voxels  = 10,                    # 3D expansion radius in voxels
    dilation_pixels  = 4,                     # 2D mask expansion before projection
    score_threshold  = 0.2,                   # GDINO confidence threshold
)

# Returns:
# result["mask_ply"]    — path to dilated 3D voxel mask (.ply)
# result["mask_2d_png"] — path to 2D segmentation mask
# result["n_masked"]    — number of active voxels
```

---

## Comparison with Original Nano3D

| | Original Nano3D | This Work |
|---|---|---|
| Edit signal | Pre-edited image (Qwen-Image or manual) | Text prompt only |
| Region mask | None — global FlowEdit | Text-guided 3D voxel mask |
| Geometry merge | `filter_edit_regions` | Voxel mask gates merge |
| SLAT merge | Coordinate-based | Cube mask gates merge |
| Conditioning | Single target image embedding | `tar − src` direction vector |
| Alpha sweep | Single output | 6 outputs at α = 0.0 … 1.0 |

---

## Built On

- [TRELLIS](https://github.com/microsoft/TRELLIS) — sparse 3D generation backbone
- [Nano3D](https://arxiv.org/abs/2510.15019) — FlowEdit integration and Voxel/Slat-Merge
- [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) — open-vocabulary detection
- [SAM](https://huggingface.co/facebook/sam-vit-base) — segment anything model
- [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) — text-conditioned 2D segmentation
