# Text-Guided 3D Editing Without a Reference Image

**DLCV Course Project** — Divya & Srihari K G

This project extends [Nano3D](https://arxiv.org/abs/2510.15019) (built on [TRELLIS](https://github.com/microsoft/TRELLIS)) to enable 3D object editing using only a **text prompt** — no pre-edited reference image required.

---

## Overview

The original Nano3D requires a manually or automatically edited 2D image to guide the 3D edit. We remove this bottleneck by automatically generating a 3D voxel mask from a text description of the region to edit. A latent direction vector (derived from a proxy image pair) drives the geometry and appearance change, while everything outside the masked region is preserved exactly.

```
Input: source image  +  text prompt ("the bulb")
Output: edited 3D mesh (GLB) at multiple edit strengths (α = 0.0 → 1.5)
```

---

## File Structure

```
Nano_3d(mask)/
├── new_pipeline.py              End-to-end pipeline (main entry point)
├── inference2.py                Original Nano3D pipeline (reference image–based)
├── inference.py                 Legacy inference script
├── app.py                       Gradio demo app
├── new_mask.py                  Standalone CLIPSeg 2D mask visualiser
├── setup.sh                     Environment setup script
├── requirements.txt
├── images/                      Sample input images (lamp, sofa)
├── assets/                      Figures used in this README
├── inference/                   Modular pipeline components
│   ├── create_mask.py           Text-guided 3D voxel mask (GDINO + SAM + ray casting)
│   ├── model_utils.py           TRELLIS method injection, VoxelProcessor, encoders
│   ├── rendering.py             Blender-based front-view renderer
│   ├── sampling.py              FlowEdit sparse-structure sampler
│   ├── voxel_encoding.py        Voxel → latent encoding utilities
│   ├── voxelization.py          Mesh → voxel grid conversion
│   ├── image_processing.py      Background removal, resizing helpers
│   └── qwen_image_edit.py       Qwen-Image automatic image editing (original approach)
├── extensions/vox2seq/          Sparse voxel-to-sequence CUDA extension
└── trellis/                     Vendored TRELLIS model code
```

---

## Installation

### 1. Base Environment

Follow the [TRELLIS installation guide](https://github.com/microsoft/TRELLIS) to set up the base conda environment, then install additional dependencies:

```bash
bash setup.sh
# or manually:
pip install -r requirements.txt
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install transformers scipy
```

### 2. Automatic Model Downloads

The mask pipeline downloads these models from Hugging Face on first run:

| Model | Used by |
|-------|---------|
| `IDEA-Research/grounding-dino-base` | `create_mask.py` |
| `facebook/sam-vit-base` | `create_mask.py` |
| `CIDAS/clipseg-rd64-refined` | `new_mask.py` only |
| `microsoft/TRELLIS-image-large` | core pipeline |

---

## Running the New Pipeline

### `new_pipeline.py` — Image + Text → Edited 3D Mesh

#### Step 1 — Configure

Edit the configuration block at the top of [new_pipeline.py](new_pipeline.py):

```python
SRC_INPUT_IMAGE_PATH = "images/lamp.jpeg"   # source image
TEXT_PROMPT          = "the bulb"           # region to edit (passed to GDINO + SAM)
OUTPUT_DIR           = "outputs/new_pipeline"

EDITING_SEED         = 1
ST_STEP              = 12    # FlowEdit aggressiveness (higher = bigger geometry change)
DILATION_VOXELS      = 10   # how far to expand the 3D mask outward (in voxels)
```

#### Step 2 — Set Proxy Direction Images

Open [new_pipeline.py](new_pipeline.py) and update the hardcoded proxy pair paths inside `run_with_mask` (lines ~102–103):

```python
thick_img  = Image.open("path/to/thick_reference.png")   # "more" direction
thin_image = Image.open("path/to/thin_reference.png")    # "less" direction
```

These two images define the edit direction in DINOv2 embedding space:
`direction = encode(thick) − encode(thin)`.  
They do **not** need to show the same object as the source; any pair that captures the intended semantic change works (e.g. a tall lamp vs a short lamp, a thick sofa vs a thin sofa).

#### Step 3 — Run

```bash
python new_pipeline.py
```

---

## Pipeline — Step by Step

### Step 1 — Generate 3D from Source Image

`run_custom` passes the source image through TRELLIS to produce a sparse voxel structure, a structured latent (SLAT), and an initial mesh saved as `src_mesh.glb`.

### Step 2 — Render Front View

The generated mesh is rendered from the front using Blender. The render (`front.png`) and camera metadata (`front_metadata.json`) are saved to `outputs/image/` and used by the ray caster in Step 3.

### Step 3 — Create 3D Voxel Mask from Text Prompt

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

**Key design decisions:**

- **Smallest box selection** — Grounding DINO is asked for the *smallest* valid box above the score threshold, not the highest-confidence one. This avoids the common failure mode where the top-scoring box covers the entire object instead of just the target part.
- **Segment on render, not source** — SAM runs on the rendered front view (geometrically aligned with the voxel grid), not the original input photo, so the 2D mask maps directly onto the 3D voxel coordinates.
- **Ray casting** — one ray per masked pixel is fired from the camera, stopping at the first occupied voxel (surface hit).
- **3D dilation** — a spherical kernel expands the surface mask to capture interior voxels that are occluded from the camera. A tight axis-aligned bounding-box (cube mask) is also computed for use in the SLAT merge step.

### Step 4 — Build Edit Direction

A direction in DINOv2 embedding space is computed from the proxy image pair:

```
direction = encode(thick_image) − encode(thin_image)
```

This captures the intended semantic change without requiring an edited version of the specific input object.

### Step 5 — Alpha-Loop Editing

For each α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5}:

1. **Interpolate conditioning:**
   ```
   cur_cond = src_cond + α × direction
   cur_cond = cur_cond × (‖src_cond‖ / ‖cur_cond‖)   # norm-preserve
   ```
2. **Sample new sparse structure** (geometry) with FlowEdit using `cur_cond`
3. **Voxel merge** — inside text mask → new geometry; outside → original source geometry
4. **Sample new SLAT** (appearance) on merged voxel coordinates
5. **SLAT merge** — inside cube mask → new appearance features; outside → original source features
6. Decode and export `edit_mesh.glb`

The norm-preserving rescale in step 1 keeps the conditioning magnitude stable across all α values, which prevents the decoder from interpreting the interpolated embedding as a lower-confidence condition.

The α sweep (including super-linear values 1.2 and 1.5) lets you explore the full range from no change to an amplified edit.

---

## Output Structure

```
outputs/new_pipeline/
├── src_mesh.glb                   source mesh
├── latent.pt                      voxel latent from TRELLIS generation
├── voxels.ply                     source voxel grid
├── image/
│   ├── front.png                  rendered front view (white background)
│   └── front_metadata.json        camera parameters for ray casting
├── mask/
│   ├── mask_2d.png                2D segmentation mask (debug)
│   ├── mask.ply                   dilated 3D voxel mask
│   └── cube_mask.ply              dense bounding-box cube mask
├── alpha_0.0/
│   ├── edit_mesh.glb              decoded output mesh
│   ├── edit_voxel.ply             raw geometry from FlowEdit
│   ├── edit_voxel_merged.ply      geometry after voxel merge
│   ├── text_mask.ply              text mask reprojected for this alpha
│   └── slat_merge_viz.ply         red=edited  green=preserved  grey=new
├── alpha_0.2/  …
├── alpha_1.0/  …
├── alpha_1.2/  …
└── alpha_1.5/  …
```

---

## Standalone 2D Mask Check (`new_mask.py`)

Before running the full pipeline, sanity-check what a text prompt selects in 2D using CLIPSeg:

```bash
python new_mask.py
# Edit IMAGE_PATH and PROMPT at the bottom of the file.
```

Outputs three files in the current directory:

| File | Content |
|------|---------|
| `comparison_plot.png` | source image side-by-side with the mask heatmap |
| `heatmap_mask.png` | continuous confidence heatmap |
| `binary_mask.png` | thresholded black-and-white mask |

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
| Edit signal | Pre-edited image (Qwen-Image or manual) | Text prompt + proxy direction pair |
| Region mask | None — global FlowEdit | Text-guided 3D voxel mask |
| Geometry merge | `filter_edit_regions` (XOR + connected components) | Voxel mask gates merge |
| SLAT merge | Coordinate-based (source overwrites target everywhere) | Cube mask gates merge (preserve outside, edit inside) |
| Conditioning | Single target image embedding interpolated to source | `src + α × (thick − thin)` direction vector |
| Alpha sweep | 6 outputs at α = 0.0 … 1.0 | 8 outputs at α = 0.0 … 1.5 |

---

## Built On

- [TRELLIS](https://github.com/microsoft/TRELLIS) — sparse 3D generation backbone
- [Nano3D](https://arxiv.org/abs/2510.15019) — FlowEdit integration and Voxel/SLAT-Merge
- [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) — open-vocabulary object detection
- [SAM](https://huggingface.co/facebook/sam-vit-base) — Segment Anything Model
- [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) — text-conditioned 2D segmentation (standalone mask check only)
