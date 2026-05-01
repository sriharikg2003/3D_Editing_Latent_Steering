import os
import argparse
import torch
import open3d as o3d
from typing import List
import imageio
from trellis.pipelines.trellis_text_to_3d import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
from trellis import pipelines
from trellis.pipelines import samplers

# Assuming 'pipeline.py' is the file containing your TrellisTextTo3DPipeline
# and it is accessible in your python path. Adjust the import as necessary.

@torch.no_grad()
def run_midway_injection_experiment(
    pipeline: TrellisTextTo3DPipeline,
    base_prompt: str,
    edit_prompt: str,
    mid_step: int = 25, 
    total_steps: int = 50,
    num_samples: int = 1,
    seed: int = 42,
    noise_blend_alpha: float = 1.0, # 1.0 = use exact feature, <1.0 = blend with noise
    cfg_strength: float = 7.5,                      # <--- ADDED: Explicit CFG Scale
    cfg_interval: tuple[float, float] = (0.0, 0.8),
    formats: List[str] = ['mesh']
) -> dict:
    """
    Executes the 3D generation experiment:
    1. Generates a base structure up to a midpoint.
    2. Extracts that intermediate structural latent.
    3. Uses it as the absolute starting point (t=1, step 0) for a new generation.
    """
    print(f"\n[Experiment Config] Base: '{base_prompt}' | Edit: '{edit_prompt}' | Mid-Step: {mid_step}")
    torch.manual_seed(seed)
    
    # 1. Get Text Conditions
    cond_base = pipeline.get_cond([base_prompt])
    cond_edit = pipeline.get_cond([edit_prompt])
    
    # 2. Setup Flow Model & Initial Pure Noise
    flow_model = pipeline.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    initial_noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
    formats = ['mesh', 'gaussian', 'radiance_field']
    decoder = pipeline.models['sparse_structure_decoder']
    # --- PHASE 1: Generate Base Structure up to 'mid_step' ---
    print(f"\n---> PHASE 1: Generating Base Structure (stopping at step {mid_step}/{total_steps})")
    base_params = {
        **pipeline.sparse_structure_sampler_params, 
        'steps': total_steps, 
        'start_step': 0, 
        'stop_step': mid_step,
        # 'cfg_strength': cfg_strength, # <--- INJECTED CFG
        # 'cfg_interval': cfg_interval  # <--- INJECTED DROP-OUT
    }
    
    midway_result = pipeline.sparse_structure_sampler.sample(
        flow_model,
        initial_noise,
        **cond_base,
        **base_params,
        verbose=True
    )
    
    # Extract the mid-way latent state
    midway_latent = midway_result.samples 

    # Optional: Blend with noise if the flow model diverges with pure structure at t=1
    if noise_blend_alpha < 1.0:
        print(f"     Blending midway feature with pure noise (alpha={noise_blend_alpha})")
        pure_noise = torch.randn_like(midway_latent)
        midway_latent = (noise_blend_alpha * midway_latent) + ((1.0 - noise_blend_alpha) * pure_noise)

    edit_cfg_strength = 0
    # --- PHASE 2: Run FULL 50 steps using Midway Latent as Step 0 ---
    print(f"\n---> PHASE 2: Running Full {total_steps} Steps with Edit Prompt")
    print(f"     Initializing Flow Transformer with Phase 1 midway feature...")
    edit_params = {
        **pipeline.sparse_structure_sampler_params, 
        'steps': total_steps, 
        'start_step': 0, 
        'stop_step': total_steps,
        'cfg_strength': edit_cfg_strength, # <--- INJECTED CFG
        'cfg_interval': cfg_interval
    }
    
    final_structure_result = pipeline.sparse_structure_sampler.sample(
        flow_model,
        midway_latent, # Injecting the midway feature as the starting point!
        **cond_edit,
        **edit_params,
        verbose=True
    )
    z_s_edited = final_structure_result.samples

    # --- PHASE 3: Decode to 3D Formats ---
    print("\n---> PHASE 3: Decoding Structured Latents into 3D Output")
    decoder = pipeline.models['sparse_structure_decoder']
    coords = torch.argwhere(decoder(z_s_edited) > 0)[:, [0, 2, 3, 4]].int()

    # slat = pipeline.sample_slat(cond_edit, coords, pipeline.slat_sampler_params)
    slat_params = {
        **pipeline.slat_sampler_params,
        'cfg_strength': cfg_strength,
        'cfg_interval': cfg_interval
    }
    slat = pipeline.sample_slat(cond_base, coords, slat_params)
    return pipeline.decode_slat(slat, formats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Trellis Mid-way Feature Injection Experiment")
    parser.add_argument("--model_path", type=str, default="microsoft/TRELLIS-text-xlarge", help="HuggingFace model path")
    parser.add_argument("--base_prompt", type=str, default="A cushioned chair", help="Initial prompt to generate structure")
    parser.add_argument("--edit_prompt", type=str, default="Increase the size of the cushion", help="Prompt used for the second full generation cycle")
    parser.add_argument("--mid_step", type=int, default=25, help="Step to stop Phase 1 and extract the feature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--alpha", type=float, default=1.0, help="Blend ratio of midway feature vs pure noise (1.0 = pure feature)")
    parser.add_argument("--output_dir", type=str, default="./experiment_outputs", help="Directory to save the resulting 3D files")
    
    args = parser.parse_args()

    # 1. Initialize Pipeline
    print(f"Loading Trellis Pipeline from {args.model_path}...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()

    # 2. Run Experiment
    outputs = run_midway_injection_experiment(
        pipeline=pipeline,
        base_prompt=args.base_prompt,
        edit_prompt=args.edit_prompt,
        mid_step=args.mid_step,
        total_steps=50,
        seed=args.seed,
        noise_blend_alpha=args.alpha,
        formats=['mesh'] # Sticking to mesh for easy saving
    )
    output_dir = "./experiment_outputs/new_cfg"
    alpha = 0.5 
    # 3. Save Output
    os.makedirs(args.output_dir, exist_ok=True)

    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(os.path.join(output_dir, f"alpha_{alpha:.1f}_gs.mp4"), video_gs, fps=30)
        
    video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(os.path.join(output_dir, f"alpha_{alpha:.1f}_rf.mp4"), video_rf, fps=30)
        
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(os.path.join(output_dir, f"alpha_{alpha:.1f}_mesh.mp4"), video_mesh, fps=30)

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(os.path.join(output_dir, f"alpha_{alpha:.1f}.glb"))

    outputs['gaussian'][0].save_ply(os.path.join(output_dir, f"alpha_{alpha:.1f}.ply"))
    
    # if 'mesh' in outputs and outputs['mesh'] is not None:
    #     # Trellis returns a list of meshes if num_samples > 1, we take the first one
    #     output_mesh = outputs['mesh'][0] 
        
    #     save_path = os.path.join(args.output_dir, f"experiment_seed{args.seed}_mid{args.mid_step}.obj")
        
    #     # Open3D requires TriangleMesh format
    #     o3d_mesh = o3d.geometry.TriangleMesh()
    #     o3d_mesh.vertices = o3d.utility.Vector3dVector(output_mesh.vertices.detach().cpu().numpy())
    #     o3d_mesh.triangles = o3d.utility.Vector3iVector(output_mesh.faces.detach().cpu().numpy())
        
    #     if hasattr(output_mesh, 'vertex_colors'):
    #         o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(output_mesh.vertex_colors.detach().cpu().numpy())

    #     o3d.io.write_triangle_mesh(save_path, o3d_mesh)
    #     print(f"\n[Success] Saved resulting mesh to: {save_path}")
    # else:
    #     print("\n[Warning] Mesh generation failed or was not returned by the pipeline.")