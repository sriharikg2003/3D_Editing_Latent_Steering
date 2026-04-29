import os
import torch
import imageio
import gc
import re
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

os.environ['SPCONV_ALGO'] = 'native'

pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()

LOAD_DIR = '/mnt/data/srihari/my_TRELLIS/LOAD'
BASE_OUTPUT_DIR = "/mnt/data/srihari/my_TRELLIS/OUTPUT_IMG_COND_AGAIN"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

PROMPT = "Girl with a black spectacles"
CFG_ARRAY = [1,5,20]


def extract_fraction(filename):
    match = re.search(r"latent_(.+)\.pt", filename)
    if match:
        return float(match.group(1))
    return 0.0

all_files = [f for f in os.listdir(LOAD_DIR) if f.endswith('.pt')]
all_files.sort(key=extract_fraction)

indices = np.linspace(0, len(all_files) - 1, 4, dtype=int)
selected_files = [all_files[i] for i in indices]

for filename in selected_files:
    latent_path = os.path.join(LOAD_DIR, filename)
    file_base = os.path.splitext(filename)[0]
    
    file_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, file_base)
    os.makedirs(file_specific_output_dir, exist_ok=True)
    
    print(f"\nProcessing File: {filename}")
    
    for cfg_strength in CFG_ARRAY:
        print(f"  Running CFG Strength: {cfg_strength}")
        
        outputs = pipeline.run_at_t(
            latent_path,
            PROMPT,
            seed=1,
            sparse_structure_sampler_params={'cfg_strength': cfg_strength}
        )

        video_data = render_utils.render_video(outputs['gaussian'][0])['color']
        video_path = os.path.join(file_specific_output_dir, f"{file_base}_cfg_{cfg_strength}.mp4")
        
        imageio.mimsave(video_path, video_data, fps=30)
        
        del outputs
        del video_data
        gc.collect()
        torch.cuda.empty_cache()

print("\nBatch processing completed.")