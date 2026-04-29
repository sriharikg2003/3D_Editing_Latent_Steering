import os
os.environ['SPCONV_ALGO'] = 'xformers'

import imageio
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

pipeline = TrellisImageTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-image-large")
pipeline.cuda()

image1 = Image.open("/mnt/data/srihari/my_TRELLIS/slim.png")
image2 = Image.open("/mnt/data/srihari/my_TRELLIS/thick.png")
test_image = Image.open("/mnt/data/srihari/my_TRELLIS/suit.avif")

alphas = [-0.5, 0.0, 0.5, 1.0, 1.5]

for alpha in alphas:
    print(f"Running Inference: ALPHA = {alpha}")
    
    outputs = pipeline.run_thickness_steer_with_strength(
        image_slim=image1,
        image_thick=image2,
        test_image=test_image,
        seed=1,
        alpha=alpha,
        strength=1.0
    )

    NAME = f"alpha_{alpha:.2f}"
    
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{NAME}_gs.mp4", video_gs, fps=30)
    
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{NAME}_mesh.mp4", video_mesh, fps=30)
    
    torch.cuda.empty_cache()

print("All 5 variations generated successfully.")