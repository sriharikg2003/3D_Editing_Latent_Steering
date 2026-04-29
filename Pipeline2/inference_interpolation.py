#!/usr/bin/env python3
"""
Interpolated 3D sword thickness generation using Nano3D.

Given a thin-sword image (source) and a thick-sword image (target),
this script generates intermediate 3D meshes by interpolating the
FlowEdit velocity delta and SLAT conditioning by strength alpha in [0, 1].

alpha=0.0 → original thin sword (no edit)
alpha=1.0 → fully thick sword (full edit)

Only the sword region is updated; the animal body is preserved via SLat-Merge.

Usage:
   CUDA_VISIBLE_DEVICES=1  python inference_interpolation.py \
        --src_image y_s.png \
        --tar_image y_b.png \
        --output_dir ./output_interp_yellow_hat \
        --editing_mode add \
        --n_steps 6
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from types import MethodType

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

from inference.image_processing import bg_to_white, resize_to_512
from inference.rendering import render_front_view
from inference.model_utils import load_sparse_structure_encoder, inject_methods

torch.set_grad_enabled(False)

# ============================================================================
# Load pipeline (once at module level)
# ============================================================================
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline = load_sparse_structure_encoder(pipeline)
pipeline = inject_methods(pipeline)
print("\nTRELLIS pipeline loaded.\n")


def compile_mp4(frame_paths, output_path, fps=2):
    """Compile a list of image paths into an MP4 using imageio."""
    try:
        import imageio
        frames = []
        for p in frame_paths:
            if os.path.exists(p):
                img = np.array(Image.open(p).convert("RGB"))
                frames.append(img)
            else:
                print(f"  [WARNING] Frame not found, skipping: {p}")
        if frames:
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"  MP4 saved: {output_path}")
        else:
            print("  [WARNING] No frames found, MP4 not created.")
    except ImportError:
        print("  [WARNING] imageio not installed. Skipping MP4 creation.")
        print("  Install with: pip install imageio[ffmpeg]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nano3D Sword Thickness Interpolation")
    parser.add_argument(
        "--src_image", type=str, default="thin.png",
        help="Path to source image (animal holding thin sword)"
    )
    parser.add_argument(
        "--tar_image", type=str, default="thick.png",
        help="Path to target image (animal holding thick sword)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output_interp",
        help="Root directory to save all outputs"
    )
    parser.add_argument(
        "--editing_mode", type=str, default="add",
        choices=["add", "remove", "replace"],
        help="Nano3D editing mode"
    )
    parser.add_argument(
        "--n_steps", type=int, default=6,
        help="Number of interpolation steps (includes alpha=0 and alpha=1)"
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(f"{output_dir}/image", exist_ok=True)

    # =========================================================================
    # STEP-0: Reconstruct source 3D mesh from thin-sword image (run once)
    # =========================================================================
    print("=" * 60)
    print("STEP-0: Source 3D reconstruction from thin-sword image")
    print("=" * 60)
    result = pipeline.run_custom(
        args.src_image,
        seed=args.seed,
        output_path=output_dir,
    )
    with torch.enable_grad():
        src_glb = postprocessing_utils.to_glb(
            result["src_mesh"]["gaussian"][0],
            result["src_mesh"]["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )
    src_mesh_path = f"{output_dir}/src_mesh.glb"
    src_glb.export(src_mesh_path)
    print(f"Source mesh saved: {src_mesh_path}")

    # =========================================================================
    # STEP-1: Render front view of source mesh
    # =========================================================================
    print("\nSTEP-1: Rendering front view of source mesh")
    render_front_view(
        file_path=src_mesh_path,
        output_dir=f"{output_dir}/image",
        output_name="front.png",
    )
    src_image_path = bg_to_white(f"{output_dir}/image/front.png")
    print(f"Front view saved: {src_image_path}")

    # =========================================================================
    # STEP-2: Prepare target (thick-sword) image
    # =========================================================================
    print("\nSTEP-2: Preparing thick-sword target image")
    tar_image_path = resize_to_512(args.tar_image, f"{output_dir}/image")
    print(f"Target image: {tar_image_path}")

    # =========================================================================
    # STEP-3: Interpolation loop over alpha values
    # =========================================================================
    alphas = np.linspace(0.0, 1.0, args.n_steps)


    print(f"\nSTEP-3: Running interpolation over {args.n_steps} steps")
    print(f"  Alpha values: {[round(a, 3) for a in alphas]}")
    print(f"  Editing mode: {args.editing_mode}")

    frame_paths = []

    for alpha in alphas:
        alpha_str = f"{alpha:.2f}"
        alpha_dir = os.path.join(output_dir, f"alpha_{alpha_str}")
        os.makedirs(f"{alpha_dir}/image", exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Interpolation alpha = {alpha_str}")
        print(f"{'='*60}")

        outputs = pipeline.run(
            src_image_path,
            tar_image_path,
            source_ply_path=f"{output_dir}/voxels.ply",
            source_voxel_latent_path=f"{output_dir}/latent.pt",
            source_slat=result["src_slat"],
            editing_mode=args.editing_mode,
            interp_alpha=float(alpha),
            seed=args.seed,
            output_path=alpha_dir,
        )

        # Save GLB
        with torch.enable_grad():
            tar_glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
            )
        glb_path = f"{alpha_dir}/edit_mesh.glb"
        tar_glb.export(glb_path)
        print(f"  GLB saved: {glb_path}")

        # Render front view for video compilation
        render_front_view(
            file_path=glb_path,
            output_dir=f"{alpha_dir}/image",
            output_name="front.png",
        )
        frame_path = f"{alpha_dir}/image/front.png"
        frame_paths.append(frame_path)
        print(f"  Front view saved: {frame_path}")

    # =========================================================================
    # STEP-4: Compile MP4 from all front-view frames
    # =========================================================================
    print(f"\nSTEP-4: Compiling interpolation video")
    mp4_path = os.path.join(output_dir, "interpolation.mp4")
    compile_mp4(frame_paths, mp4_path, fps=2)

    # Also save a side-by-side comparison image of all alpha frames
    frames_loaded = []
    for p in frame_paths:
        if os.path.exists(p):
            frames_loaded.append(Image.open(p).convert("RGB").resize((256, 256)))
    if frames_loaded:
        comparison = Image.new("RGB", (256 * len(frames_loaded), 256))
        for i, img in enumerate(frames_loaded):
            comparison.paste(img, (i * 256, 0))
        comparison_path = os.path.join(output_dir, "comparison.png")
        comparison.save(comparison_path)
        print(f"Comparison image saved: {comparison_path}")

    print("\n" + "=" * 60)
    print("DONE! Outputs saved to:", output_dir)
    print(f"  - src_mesh.glb          : original thin-sword 3D mesh")
    print(f"  - alpha_X.XX/edit_mesh.glb : interpolated meshes for each alpha")
    print(f"  - interpolation.mp4     : video showing thickness progression")
    print(f"  - comparison.png        : side-by-side front views")
    print("=" * 60)


