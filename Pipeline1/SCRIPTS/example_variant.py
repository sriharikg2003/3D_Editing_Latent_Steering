
import os
os.environ['SPCONV_ALGO'] = 'native'

import imageio
import numpy as np


import torch
import open3d as o3d
import trimesh
path = "/mnt/data/srihari/my_TRELLIS/chair.obj"
t_mesh = trimesh.load(path)
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


# Manually build the Open3D mesh for the pipeline
base_mesh = o3d.geometry.TriangleMesh()
base_mesh.vertices = o3d.utility.Vector3dVector(np.array(t_mesh.vertices))
base_mesh.triangles = o3d.utility.Vector3iVector(np.array(t_mesh.faces))

if not base_mesh.has_vertices():
    raise ValueError(f"Failed to load vertices from {path}. The file might be empty.")
# --- REPAIR LOGIC END ---



# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()

# --- REPAIR LOGIC START ---
# Load with trimesh to bypass "Wrong magic number" PLY errors

for STRENGTH in [0,1,3,4,7,10,12,20]:

    # Run the pipeline
    outputs = pipeline.run_variant(
        base_mesh,
        "red metallic",
        seed=1,
        # slat_sampler_params={
        #     "steps": 25,
        #     "cfg_strength": STRENGTH,
        # },
    )


    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"VARIANT_{STRENGTH}.mp4", video, fps=30)
# video = render_utils.render_video(outputs['radiance_field'][0])['color']
# imageio.mimsave("sample_rf.mp4", video, fps=30)
# video = render_utils.render_video(outputs['mesh'][0])['normal']
# imageio.mimsave("sample_mesh.mp4", video, fps=30)

# # Save the final mesh properly using trimesh to avoid future header issues
# mesh_data = outputs['mesh'][0]
# final_export = trimesh.Trimesh(
#     vertices=mesh_data.vertices.cpu().numpy(),
#     faces=mesh_data.faces.cpu().numpy(),
#     process=False
# )

# if mesh_data.vertex_attrs is not None:
#     colors = mesh_data.vertex_attrs.cpu().numpy()[:, :3]
#     final_export.visual.vertex_colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

# final_export.export("just_the_mesh_fixed.obj")