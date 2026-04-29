"""
bootstrap_dataset.py
--------------------
1. Generate 3D objects from text prompts using text pipeline
2. Render each object to a clean image
3. Save image + mesh for use in generate_pairs.py
"""

import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

import torch
import imageio
import numpy as np
from PIL import Image
from pathlib import Path
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

MODEL_PATH = "/mnt/data/srihari/MODELS/TRELLIS-text-xlarge"
OUT_DIR    = "BOOTSTRAP"

PROMPTS = [
    # Chairs
    "a simple wooden dining chair with four legs",
    "a modern plastic chair with smooth surface",
    "a metal folding chair with thin legs",
    "a cushioned armchair with fabric upholstery",
    "a wooden stool with three legs",
    # Tables
    "a simple wooden coffee table with four legs",
    "a round metal side table",
    "a rectangular dining table made of oak wood",
    # Vases / simple objects
    "a ceramic vase with smooth round body",
    "a tall glass bottle with narrow neck",
    "a simple clay pot with handles",
    "a round decorative bowl",
    # Shoes
    "a simple white sneaker shoe",
    "a leather boot with thick sole",
    "a sandal with flat sole",
    # Lamps
    "a table lamp with round base",
    "a simple floor lamp with thin pole",
    # Misc
    "a wooden barrel",
    "a simple backpack",
    "a round clock",
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pipeline = TrellisTextTo3DPipeline.from_pretrained(MODEL_PATH)
    pipeline.cuda()

    for idx, prompt in enumerate(PROMPTS):
        out_dir = Path(f"{OUT_DIR}/obj_{idx:03d}")
        out_dir.mkdir(exist_ok=True)

        print(f"\n[{idx+1}/{len(PROMPTS)}] {prompt}")

        # Generate
        output = pipeline.run(
            prompt, seed=42,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 12, "cfg_strength": 7.5},
        )

        # Save front-view render as image (this becomes input to image pipeline)
        video = render_utils.render_video(output['gaussian'][0], num_frames=60)['color']
        front_frame = video[0]   # frame 0 = front view
        Image.fromarray(front_frame).save(str(out_dir / "image.png"))

        # Save mesh as GLB
        glb = postprocessing_utils.to_glb(
            output['gaussian'][0],
            output['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb.export(str(out_dir / "mesh.glb"))

        # Save prompt for reference
        (out_dir / "prompt.txt").write_text(prompt)

        print(f"  Saved to {out_dir}")

    print(f"\nDone. {len(PROMPTS)} objects in {OUT_DIR}/")

if __name__ == "__main__":
    main()