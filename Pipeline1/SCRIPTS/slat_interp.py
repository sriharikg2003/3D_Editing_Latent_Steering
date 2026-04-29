
import os
import torch
import imageio
import trimesh
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

os.environ['SPCONV_ALGO'] = 'native'
FILE_NAME = "LAMP"
pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()

outputs = pipeline.run(
    "a long wide lamp",
    seed=1,
)

video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(f"{FILE_NAME}.mp4", video, fps=30)
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
export_mesh.export(f"{FILE_NAME}.obj")

glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)







# import os
# os.environ['SPCONV_ALGO'] = 'native'

# import imageio
# import numpy as np
# import torch
# import open3d as o3d
# import trimesh

# path = f"{FILE_NAME}.obj"
# t_mesh = trimesh.load(path)
# from trellis.pipelines import TrellisTextTo3DPipeline
# from trellis.utils import render_utils

# base_mesh = o3d.geometry.TriangleMesh()
# base_mesh.vertices = o3d.utility.Vector3dVector(np.array(t_mesh.vertices))
# base_mesh.triangles = o3d.utility.Vector3iVector(np.array(t_mesh.faces))

# if not base_mesh.has_vertices():
#     raise ValueError(f"Failed to load vertices from {path}.")

# pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
# pipeline.cuda()

# slat1, slat2 = pipeline.sample_slats_for_interp(base_mesh, prompt1="red hair", prompt2="yellow hair", seed=1)

# for STRENGTH in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
#     outputs = pipeline.decode_slat_interp(slat1, slat2, alpha=STRENGTH)
#     video = render_utils.render_video(outputs['gaussian'][0])['color']
#     imageio.mimsave(f"slat_interp_diff_{STRENGTH}.mp4", video, fps=30)