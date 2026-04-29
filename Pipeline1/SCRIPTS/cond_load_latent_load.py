import os
import torch
import imageio
import gc
import re
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

os.environ['SPCONV_ALGO'] = 'native'



# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-image-large")
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

indices = np.linspace(0, len(all_files) - 1, 8, dtype=int)
selected_files = [all_files[i] for i in indices]

for filename in selected_files:
    latent_path = os.path.join(LOAD_DIR, filename)
    file_base = os.path.splitext(filename)[0]
    
    file_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, file_base)
    os.makedirs(file_specific_output_dir, exist_ok=True)
    
    print(f"\nProcessing File: {filename}")
    
    for cfg_strength in CFG_ARRAY:
        print(f"  Running CFG Strength: {cfg_strength}")
        PATH_SAVE = f"/mnt/data/srihari/my_TRELLIS/LOAD_COORDS/{str(cfg_strength)}/{filename[:-3]}/coords.pt"
        outputs = pipeline.run_at_t_load_cond_and_latent(
            PATH_SAVE,
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