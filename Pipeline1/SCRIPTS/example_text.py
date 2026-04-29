


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


# noise = torch.load('')

outputs = pipeline.run(
    "Generate a very THICK structure.",
    seed=1,
)

video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("ball.mp4", video, fps=30)
# video = render_utils.render_video(outputs['radiance_field'][0])['color']
# imageio.mimsave("sample_rf.mp4", video, fps=30)
# video = render_utils.render_video(outputs['mesh'][0])['normal']
# imageio.mimsave("sample_mesh.mp4", video, fps=30)

mesh_data = outputs['mesh'][0]
vertices = mesh_data.vertices.cpu().numpy()
faces = mesh_data.faces.cpu().numpy()
export_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

if mesh_data.vertex_attrs is not None:
    colors = mesh_data.vertex_attrs.cpu().numpy()[:, :3]
    colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    export_mesh.visual.vertex_colors = colors


# FOR editing or texture transfer
export_mesh.export("ball.obj")

glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export("ball.glb")

outputs['gaussian'][0].save_ply("ball.ply")