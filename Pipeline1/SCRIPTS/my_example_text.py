OUTPUT_FOLDER = "ALL_OUTPUTS"

import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()

PROMPTS = [
  "Two high-tensile steel chains tightly interlocked in a complex Gordian knot, each link clearly separated with visible air gaps.",
  "An intricate Victorian birdcage made of hair-thin golden wires with a small mechanical bird inside.",
  "A transparent magnifying glass held by a hand, showing a realistically refracted and magnified ladybug through the lens.",
  "A giant skyscraper constructed entirely of individual, 1x1 LEGO bricks where every single connector peg is geometrically defined.",
  "A 3D realization of an M.C. Escher Penrose Triangle sculpture that maintains its impossible geometry from all viewing angles.",
  "A highly detailed wire-mesh chair where the wires are thinner than a single voxel, forming a semi-transparent lattice.",
  "A stack of five translucent glass prisms reflecting and refracting a rainbow spectrum onto a white surface.",
  "A hyper-realistic dandelion with hundreds of individual thin filaments radiating from the center seed head.",
  "A ship inside a narrow-necked glass bottle, where the glass thickness and the internal wooden ship details are both fully resolved.",
  "A dense clump of long, wet fur or hair, where every strand is individually modeled rather than represented as a solid volume."
]

for idx , PROMPT in enumerate(PROMPTS):


    outputs = pipeline.run(
        PROMPT,
        seed=1,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes
    # Render the outputs
    os.mkdir(f"{OUTPUT_FOLDER}/sample_{idx}")
    SAVE_PATH = f"{OUTPUT_FOLDER}/sample_{idx}"
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{SAVE_PATH}/sample_gs.mp4", video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(f"{SAVE_PATH}/sample_rf.mp4", video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{SAVE_PATH}/sample_mesh.mp4", video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(f"{SAVE_PATH}/sample.glb")

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(f"{SAVE_PATH}/sample.ply")
