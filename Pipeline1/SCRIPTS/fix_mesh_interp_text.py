import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils
import imageio, os

pipeline = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()


# # Generate a base mesh first
# output = pipeline.run(
#     "a wooden chair",
#     seed=42,
#     sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
#     slat_sampler_params={"steps": 12, "cfg_strength": 7.5},
# )

# # Save GLB
# from trellis.utils import postprocessing_utils
# glb = postprocessing_utils.to_glb(
#     output['gaussian'][0],
#     output['mesh'][0],
#     simplify=0.95,
#     texture_size=1024,
# )
# glb.export("base_chair.glb")

# Load it back as open3d mesh
mesh = o3d.io.read_triangle_mesh("base_chair.glb")
print("Vertices:", len(mesh.vertices))




pairs = [
    (
        "a rustic wooden farmhouse chair with visible wood grain, natural brown oak texture, rough hewn surface, warm earthy tones",
        "a sleek modern chrome metal chair with highly polished reflective steel surface, cold silver metallic sheen, industrial finish"
    ),
    (
        "a heavily weathered antique chair with peeling paint, cracked surface, aged patina, distressed brown and white finish",
        "a brand new glossy plastic chair in vivid solid red, smooth uniform surface, shiny injection molded finish"
    ),
    (
        "a soft plush velvet armchair in deep royal purple, thick padded cushions, luxurious fabric texture, ornate carved wooden legs",
        "a minimalist concrete chair with raw cement surface, coarse aggregate texture, cold grey industrial brutalist aesthetic"
    ),
]

for i, (p1, p2) in enumerate(pairs):
    results = pipeline.my_run_variant_slerp(
        mesh=mesh,
        prompt1=p1,
        prompt2=p2,
        num_interpolations=10,
        formats=['gaussian']
    )
    os.makedirs(f"SLERP_INTERP/pair_{i}", exist_ok=True)
    for j, r in enumerate(results):
        video = render_utils.render_video(r['gaussian'][0])['color']
        imageio.mimsave(f"SLERP_INTERP/pair_{i}/step_{j:02d}.mp4", video, fps=30)
    print(f"Done pair {i}: {p1} -> {p2}")