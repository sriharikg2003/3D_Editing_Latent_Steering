# Pipeline 2 — 3D Latent Interpolation via TRELLIS

Pipeline 2 enables smooth morphing between two 3D objects by interpolating in the latent space of the [TRELLIS](https://github.com/microsoft/TRELLIS) image-to-3D framework. Given a source and target image, it reconstructs both objects, then generates a sequence of intermediate meshes ranging from one to the other using flow-based velocity interpolation.

---

## How It Works

The pipeline decouples **geometry** (sparse voxel structure) from **appearance** (SLAT tokens), editing each independently:

1. **Source reconstruction** — TRELLIS encodes the source image into a sparse voxel latent and a SLAT (Structured Latent Appearance Token) tensor.
2. **FlowEdit velocity delta** — At each interpolation step α, the denoising velocity vectors for the source and target conditions are computed and blended: `ΔV = α · (V_tar − V_src)`. This steers the sparse structure latent toward the target geometry.
3. **SLAT interpolation** — Appearance is interpolated linearly: `(1−α)·slat_src + α·slat_tar`, preserving source texture at low α values.
4. **Mesh decoding** — The edited latent is decoded back to a mesh (`.glb`) and optionally rendered.

---

## Directory Structure

```
Pipeline2/
├── inference_interpolation.py   # Main entry point
├── render_interpolation.py      # Multi-view rendering of all alpha steps
├── requirements.txt
├── inference/
│   ├── model_utils.py           # Encoder loading, pipeline injection
│   ├── sampling.py              # FlowEdit-based interpolated sampling
│   ├── rendering.py             # Blender rendering (front view, multi-view)
│   ├── image_processing.py      # Background removal, resizing
│   ├── voxelization.py          # Mesh → voxel features (DINOv2)
│   ├── voxel_encoding.py        # VAE encode/decode for voxel grids
│   └── qwen_image_edit.py       # Optional Qwen image editing integration
├── trellis/                     # TRELLIS model framework (sparse flow, VAE, renderers)
├── extensions/vox2seq/          # CUDA extension for Hilbert/Z-order voxel ordering
└── *.png                        # Sample image pairs (thin/thick, hats, carrots, etc.)
```

---

## Requirements

### Hardware
- GPU with **≥ 24 GB VRAM** (tested on A100/H100)
- Blender installed and on `PATH` (for rendering steps)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:

| Category | Packages |
|----------|----------|
| Deep learning | `torch==2.4.0+cu118`, `torchvision==0.19.0`, `diffusers==0.34.0`, `transformers`, `accelerate` |
| Sparse 3D | `spconv-cu118`, `kaolin` |
| 3D processing | `open3d`, `trimesh`, `pymeshfix`, `pysdf`, `xatlas`, `pyvista`, `plyfile` |
| Rendering | `bpy` (Blender Python), `imageio[ffmpeg]` |
| Vision | `opencv-python` |

Build the CUDA voxel-sequence extension before running:

```bash
cd extensions/vox2seq
python setup.py install
```

---

## Pretrained Models

Downloaded automatically from HuggingFace on first run:

| Model | HuggingFace ID | Role |
|-------|---------------|------|
| TRELLIS (full pipeline) | `microsoft/TRELLIS-image-large` | Image → 3D reconstruction |
| Sparse structure encoder | `ss_enc_conv3d_16l8_fp16.safetensors` | Voxel grid → latent |
| SLAT encoder | `slat_enc_swin8_B_64l8_fp16.safetensors` | Image features → SLAT tokens |
| DINOv2 | `dinov2_vitg14` | Multi-view feature extraction |

---

## Usage

### Step 1 — Run interpolation

```bash
CUDA_VISIBLE_DEVICES=0 python inference_interpolation.py \
    --src_image thin.png \
    --tar_image thick.png \
    --output_dir ./output_interp \
    --editing_mode add \
    --n_steps 6
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--src_image` | — | Source object image (starting state) |
| `--tar_image` | — | Target object image (ending state) |
| `--output_dir` | — | Root output directory |
| `--editing_mode` | `add` | `add` · `remove` · `replace` (controls SLAT merging strategy) |
| `--n_steps` | `6` | Number of interpolation steps (number of α values between 0 and 1) |
| `--seed` | `1` | Random seed |

The script runs four internal stages:

| Stage | Output |
|-------|--------|
| **0** — Source reconstruction | `voxels.ply`, `latent.pt`, `src_mesh.glb` |
| **1** — Front-view render of source | `front.png` |
| **2** — Target image preparation | `edit_512.png` |
| **3** — Interpolation loop | `alpha_X.XX/edit_mesh.glb`, `alpha_X.XX/image/front.png` |
| **4** — Compile outputs | `interpolation.mp4`, `comparison.png` |

### Step 2 — Multi-view rendering (optional)

```bash
python render_interpolation.py \
    --interp_dir ./output_interp \
    --engine BLENDER_EEVEE \
    --resolution 512
```

For each alpha step this produces a 360° rotating video and 10 fixed-angle snapshots. It also builds per-angle side-by-side comparison strips across all alpha values.

| Output | Description |
|--------|-------------|
| `alpha_X.XX/renders/rotating.mp4` | 36-frame 360° rotation at 12 fps |
| `alpha_X.XX/renders/snapshot_*.png` | 10 fixed-angle snapshots |
| `comparison/compare_angle_*.png` | Side-by-side strip at one angle, all alphas |
| `comparison/comparison_all_angles.mp4` | Video cycling through all angles |

---

## Editing Modes

| Mode | Behavior |
|------|----------|
| `add` | Merges source appearance into regions untouched by the edit |
| `remove` | Inverts the edit direction (removes a feature) |
| `replace` | Restricts edits to a masked region only |

---

## Sample Image Pairs

Pre-included pairs for quick testing:

| Pair | Source | Target |
|------|--------|--------|
| Sword thickness | `thin.png` | `thick.png` |
| Cylinder thickness | `thin_cyl.png` | `thick_cyl.png` |
| Carrot | `carrot_thin.png` | `carrot_thick.png` |
| Yellow object size | `y_s.png` | `y_b.png` |
| Hat size | `small_hat.png` | `big_hat.png` |

---

## Technical Details

### Sparse structure sampling (`inference/sampling.py`)

At each denoising timestep `t`, the flow velocity is steered by blending source and target predictions:

```
z_t  = (1 − t)·z_src + t·ε          # forward diffusion
ΔV   = α · (V_tar − V_src)           # velocity delta
z_edit ← z_edit + Δt · ΔV            # guided update
```

Key hyperparameters:

| Parameter | Default | Role |
|-----------|---------|------|
| `n_avg` | `5` | Noise samples averaged per timestep |
| `st_step` | `12` | Timestep at which editing begins |
| `src_cfg` | `1.5` | CFG scale for source condition |
| `tar_cfg` | `5.5` | CFG scale for target condition |

### Voxel processing (`inference/voxel_encoding.py`)

- Voxel grid resolution: **64³**
- Sparse structure latent: encoded by a 16-layer 3D CNN
- Noise filtering: connected-component analysis removes components smaller than 250 voxels

### Rendering (`inference/rendering.py`)

- Camera: Hammersley-sampled poses, FOV 40°, radius 2.0
- Lights: key (1 200 W point), fill (8 000 W area), back (800 W area)
- Engines: `CYCLES` (photo-realistic, 64–128 samples) or `BLENDER_EEVEE` (fast preview)

---

## Citation

This pipeline builds on TRELLIS:

```bibtex
@article{xiang2024trellis,
  title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
  author  = {Xiang, Jianfeng and others},
  journal = {arXiv},
  year    = {2024}
}
```
