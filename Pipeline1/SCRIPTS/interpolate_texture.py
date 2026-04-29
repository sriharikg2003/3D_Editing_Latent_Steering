INPUT_PLY_FILE = "/mnt/data/srihari/my_TRELLIS/assets/T.ply"

import os
# os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

import imageio
import numpy as np
import open3d as o3d

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


# Load pipeline
pipeline = TrellisTextTo3DPipeline.from_pretrained(
    "/mnt/data/srihari/MODELS/TRELLIS-text-xlarge"
)
pipeline.cuda()


# Load mesh
base_mesh = o3d.io.read_triangle_mesh(INPUT_PLY_FILE)


# Run interpolation pipeline
outputs_list = pipeline.my_run_variant(
    base_mesh,
    "matte white ceramic surface",
    "shiny polished gold metal",
    num_interpolations=8,
    seed=1,
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
)


# Render each interpolation
for i, outputs in enumerate(outputs_list):

    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']

    video = [
        np.concatenate([frame_gs, frame_mesh], axis=1)
        for frame_gs, frame_mesh in zip(video_gs, video_mesh)
    ]

    imageio.mimsave(f"/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/T_{i}.mp4", video, fps=30)