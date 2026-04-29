
import argparse
parser = argparse.ArgumentParser(description="A script to process images using DINOv2")
parser.add_argument("--alpha")
args = parser.parse_args()
alpha = float(args.alpha)

print(f"ALPHA = {alpha} , {type(alpha)}")
NAME = f"{alpha:.2f}_{1-alpha:.2f}"
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
pipeline = TrellisImageTo3DPipeline.from_pretrained(" /mnt/data/srihari/my_TRELLIS/TRELLIS_MODEL/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image1 = Image.open("/datafrom_146/srihari/my_TRELLIS/assets/chair1.webp")
image2 = Image.open("/datafrom_146/srihari/my_TRELLIS/assets/chair2.webp")


outputs = pipeline.run_sparse_interp(
    image1 = image1,
    image2 = image2,
    seed=1,
    alpha=alpha

)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(f"{NAME}_gs.mp4", video, fps=30)
# video = render_utils.render_video(outputs['radiance_field'][0])['color']
# imageio.mimsave(f"{NAME}_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave(f"{NAME}_mesh.mp4", video, fps=30)

# # GLB files can be extracted from the outputs
# glb = postprocessing_utils.to_glb(
#     outputs['gaussian'][0],
#     outputs['mesh'][0],
#     # Optional parameters
#     simplify=0.95,          # Ratio of triangles to remove in the simplification process
#     texture_size=1024,      # Size of the texture used for the GLB
# )
# glb.export(f"{NAME}.glb")

# # Save Gaussians as PLY files
# outputs['gaussian'][0].save_ply(f"{NAME}.ply")
