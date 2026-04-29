import os
os.environ['SPCONV_ALGO'] = 'native'

import imageio
from PIL import Image
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

pipeline = TrellisImageTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-image-large")
pipeline.cuda()

image = Image.open("/mnt/data/srihari/my_TRELLIS/9b6af152ea5d5495be80bb095c32e41f2928732a6a577632971980a3aefca4e2.png")

alphas = torch.linspace(0, 1, 20).tolist()
crop_sizes = torch.linspace(4, 37, 10).long().tolist()
base_output_dir = "/mnt/data/srihari/my_TRELLIS/patched_steering_chair"
os.makedirs(base_output_dir, exist_ok=True)

for crop_size in crop_sizes:
    crop_dir = os.path.join(base_output_dir, f"crop_{crop_size:02d}")
    os.makedirs(crop_dir, exist_ok=True)
    
    for i, alpha in enumerate(alphas):
        print(f"Crop: {crop_size:02d}/37 | Alpha: {alpha:.3f} (Sample {i+1}/20)")
        
        outputs = pipeline.run_load_dino_vec(
            image,
            seed=1,
            alpha=float(alpha),
            crop_size=int(crop_size)
        )

        video = render_utils.render_video(outputs['gaussian'][0])['color']
        filename = os.path.join(crop_dir, f"alpha_{alpha:.3f}.mp4")
        imageio.mimsave(filename, video, fps=30)

print(f"Sweep finished. All 200 videos saved in {base_output_dir}")