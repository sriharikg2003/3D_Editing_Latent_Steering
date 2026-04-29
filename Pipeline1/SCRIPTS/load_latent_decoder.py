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

noise_dir = "/mnt/data/srihari/my_TRELLIS/LOAD"
base_output_dir = "/mnt/data/srihari/my_TRELLIS/OUTPUT"
os.makedirs(base_output_dir, exist_ok=True)

prompts = [
    "Very thick object"]


# /mnt/data/srihari/my_TRELLIS/LOAD/latent_0.1111111111111112.pt




def extract_fraction(filename):
    match = re.search(r"latent_(.+)\.pt", filename)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0

all_files = [f for f in os.listdir(noise_dir) if f.endswith('.pt')]
all_files.sort(key=extract_fraction)

num_samples = 5
indices = np.linspace(0, len(all_files) - 1, num_samples).astype(int)
selected_files = [all_files[i] for i in indices]

print(f"Total files found: {len(all_files)}")
print(f"Files selected for processing: {selected_files}")

for p_idx, prompt in enumerate(prompts):
    folder_name = f"prompt_{p_idx+1}_{prompt[:20].replace(' ', '_').strip()}"
    current_output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(current_output_dir, exist_ok=True)
    print(prompt)
    print(f"\n--- Running Prompt {p_idx+1}/5 ---")

    for fname in selected_files:
        noise_path = os.path.join(noise_dir, fname)
        print(f"  Processing: {fname}")

        outputs = pipeline.run_at_t(
            noise_path,
            prompt,
            seed=1,
            sparse_structure_sampler_params={'cfg_strength': 50}
        )

        base_name = os.path.splitext(fname)[0]
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(
            os.path.join(current_output_dir, f"{base_name}.mp4"),
            video,
            fps=30
        )
        
        del outputs
        del video
        gc.collect()
        torch.cuda.empty_cache()

print("\nAll tasks completed.")