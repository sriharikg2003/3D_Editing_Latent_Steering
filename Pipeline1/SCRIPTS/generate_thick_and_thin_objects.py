import os
import torch
import imageio
import trimesh
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

os.environ['SPCONV_ALGO'] = 'native'

pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()

categories = {
    "thin_vase": "A very thin, elegant ceramic vase.",
    "thick_vase": "A very thick, chunky handcrafted vase."
}

num_samples = 20
output_base_folder = "generated_vases"
os.makedirs(output_base_folder, exist_ok=True)

for label, prompt in categories.items():
    category_folder = os.path.join(output_base_folder, label)
    os.makedirs(category_folder, exist_ok=True)
    
    for i in range(num_samples):
        seed = i + 100 
        
        outputs = pipeline.run_to_generate_data(
            prompt=prompt,
            path = f"{category_folder}/{label}_{i}.pt" ,
            seed=seed,
        )

        video = render_utils.render_video(outputs['gaussian'][0])['color']
        video_path = os.path.join(category_folder, f"{label}_{i}.mp4")
        imageio.mimsave(video_path, video, fps=30)

      
      