import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import numpy as np
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()

# Load mesh to make variants
base_mesh = o3d.io.read_triangle_mesh("assets/T.ply")
PROMPTS = [
    "a shiny polished gold surface with rich metallic reflections",
    "a smooth brushed silver metal texture with subtle highlights",
    "natural wooden texture with visible grain patterns and warm brown tones",
    "dense green leafy texture with fresh leaves and natural foliage details"
]
for idx , prompt in enumerate(PROMPTS):
    # Run the pipeline
    outputs = pipeline.run_variant(
        base_mesh,
        prompt,
        seed=1,
        # Optional parameters
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    outputs['gaussian'][0].save_ply(f"/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/variant_{idx}.ply")

