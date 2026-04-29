import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image = Image.open("/mnt/data/srihari/my_TRELLIS/husky.png")
# Run the pipeline
OUT_FOLDER = f"ALL_OUTPUTS/Added_noise_latent_move"
os.makedirs(OUT_FOLDER , exist_ok = True)

slat ,noise =  pipeline.sample_slat_and_noise(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },

)
import numpy as np
alphas = np.linspace(1,10,10)

for alpha in alphas:
    outputs = pipeline.run_move_in_latent(
        seed=1,
        alpha=alpha,
        slat=slat, 
        my_noise=noise,
   
   
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    NAME = f"{alpha:.2f}"
    # Render Gaussian
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{OUT_FOLDER}/{NAME}_gs.mp4", video, fps=30)

    # Render Mesh
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{OUT_FOLDER}/{NAME}_mesh.mp4", video, fps=30)

    print(f"Finished alpha = {alpha}")


