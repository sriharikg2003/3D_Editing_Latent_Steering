import os
import argparse
import torch
import imageio
from typing import List

# Assuming these are available in your environment
from trellis.pipelines.trellis_text_to_3d import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

@torch.no_grad()
def run_midway_injection_experiment(
    pipeline: TrellisTextTo3DPipeline,
    base_prompt: str,
    edit_prompt: str,
    mid_step: int = 25, 
    total_steps: int = 50,
    num_samples: int = 1,
    seed: int = 42,
    noise_blend_alpha: float = 1.0, 
    cfg_strength: float = 7.5,                      
    cfg_interval: tuple[float, float] = (0.0, 0.8),
    formats: List[str] = ['mesh', 'gaussian', 'radiance_field']
) -> dict:
    """
    Executes the 3D generation experiment:
    1. Generates a base structure up to a midpoint.
    2. Extracts that intermediate structural latent.
    3. Uses it as the absolute starting point (t=1, step 0) for a new generation.
    """
    print(f"\n[Experiment Config] Base: '{base_prompt}' | Edit: '{edit_prompt}' | Mid-Step: {mid_step} | CFG: {cfg_strength}")
    torch.manual_seed(seed)
    
    # 1. Get Text Conditions
    cond_base = pipeline.get_cond([base_prompt])
    cond_edit = pipeline.get_cond([edit_prompt])
    
    # 2. Setup Flow Model & Initial Pure Noise
    flow_model = pipeline.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    initial_noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
    
    # --- PHASE 1: Generate Base Structure up to 'mid_step' ---
    print(f"\n---> PHASE 1: Generating Base Structure (stopping at step {mid_step}/{total_steps})")
    base_params = {
        **pipeline.sparse_structure_sampler_params, 
        'steps': total_steps, 
        'start_step': 0, 
        'stop_step': mid_step,
    }
    
    midway_result = pipeline.sparse_structure_sampler.sample(
        flow_model,
        initial_noise,
        **cond_base,
        **base_params,
        verbose=False # Set to False to reduce terminal spam in loops
    )
    
    # Extract the mid-way latent state
    midway_latent = midway_result.samples 

    # Optional: Blend with noise if the flow model diverges with pure structure at t=1
    if noise_blend_alpha < 1.0:
        print(f"     Blending midway feature with pure noise (alpha={noise_blend_alpha})")
        pure_noise = torch.randn_like(midway_latent)
        midway_latent = (noise_blend_alpha * midway_latent) + ((1.0 - noise_blend_alpha) * pure_noise)

    edit_cfg_strength = 0 # As per original script logic
    
    # --- PHASE 2: Run FULL steps using Midway Latent as Step 0 ---
    print(f"\n---> PHASE 2: Running Full {total_steps} Steps with Edit Prompt")
    edit_params = {
        **pipeline.sparse_structure_sampler_params, 
        'steps': total_steps, 
        'start_step': 0, 
        'stop_step': total_steps,
        'cfg_strength': edit_cfg_strength, 
        'cfg_interval': cfg_interval
    }
    
    final_structure_result = pipeline.sparse_structure_sampler.sample(
        flow_model,
        midway_latent, 
        **cond_edit,
        **edit_params,
        verbose=False
    )
    z_s_edited = final_structure_result.samples

    # --- PHASE 3: Decode to 3D Formats ---
    print("\n---> PHASE 3: Decoding Structured Latents into 3D Output")
    decoder = pipeline.models['sparse_structure_decoder']
    coords = torch.argwhere(decoder(z_s_edited) > 0)[:, [0, 2, 3, 4]].int()

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
    parser.add_argument("--base_prompt", type=str, default="A girl's face", help="Initial prompt to generate structure")
    parser.add_argument("--edit_prompt", type=str, default="add a spectacle", help="Prompt used for the second full generation cycle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--alpha", type=float, default=1.0, help="Blend ratio of midway feature vs pure noise (1.0 = pure feature)")
    parser.add_argument("--output_dir", type=str, default="./experiment_outputs1", help="Directory to save the resulting 3D files")
    
    # --- ADDED: Accepts lists of values for grid search ---
    parser.add_argument("--mid_steps", type=int, nargs='+', default=[20, 25, 30, 35,40], help="List of mid steps to test")
    parser.add_argument("--cfg_strengths", type=float, nargs='+', default=[1, 5.0, 7.5, 10.0, 20], help="List of CFG strengths to test")
    parser.add_argument("--frame_time", type=float, default=1.0, help="Time in seconds to extract the frame (e.g., 2.0s)")
    
    args = parser.parse_args()

    # 1. Initialize Pipeline
    print(f"Loading Trellis Pipeline from {args.model_path}...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()

    # Create the output directory once
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Iterate over all combinations of mid_step and cfg_strength
    for mid_step in args.mid_steps:
        for cfg in args.cfg_strengths:
            print(f"\n{'='*50}")
            print(f"Starting run: Mid-Step = {mid_step} | CFG = {cfg}")
            print(f"{'='*50}")
            
            outputs = run_midway_injection_experiment(
                pipeline=pipeline,
                base_prompt=args.base_prompt,
                edit_prompt=args.edit_prompt,
                mid_step=mid_step,
                cfg_strength=cfg,
                total_steps=50,
                seed=args.seed,
                noise_blend_alpha=args.alpha,
                formats=['mesh', 'gaussian', 'radiance_field'] # Required for GS/RF outputs below
            )
            
            # 3. Save Outputs with dynamic filenames
            prefix = f"mid{mid_step}_cfg{cfg:.1f}_alpha{args.alpha:.1f}"
            
            # Videos
            print(f"Saving videos for {prefix}...")
            video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(os.path.join(args.output_dir, f"{prefix}_gs.mp4"), video_gs, fps=30)
                
            video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
            imageio.mimsave(os.path.join(args.output_dir, f"{prefix}_rf.mp4"), video_rf, fps=30)
                
            video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(os.path.join(args.output_dir, f"{prefix}_mesh.mp4"), video_mesh, fps=30)

            # 3D Models
            print(f"Exporting 3D models for {prefix}...")
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(os.path.join(args.output_dir, f"{prefix}.glb"))
            outputs['gaussian'][0].save_ply(os.path.join(args.output_dir, f"{prefix}.ply"))
            
            print(f"Finished run for Mid-Step = {mid_step} | CFG = {cfg}")

    print("\n[Success] All combinations completed!")

    print("\nCompiling summary grid...")
    rows = len(args.mid_steps)
    cols = len(args.cfg_strengths)
    
    # Get dimensions from the first frame to setup the canvas
    sample_frame = next(iter(extracted_frames.values()))
    h, w, c = sample_frame.shape
    
    # Create a blank black canvas for the grid
    grid_img = Image.new('RGB', (w * cols, h * rows), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid_img)
    
    for r_idx, mid_step in enumerate(args.mid_steps):
        for c_idx, cfg in enumerate(args.cfg_strengths):
            frame_array = extracted_frames[(mid_step, cfg)]
            
            # Convert numpy array to PIL Image
            img = Image.fromarray(frame_array)
            
            # Calculate position
            x_offset = c_idx * w
            y_offset = r_idx * h
            
            # Paste image into the grid
            grid_img.paste(img, (x_offset, y_offset))
            
            # Draw text label over the image
            label = f"Mid: {mid_step} | CFG: {cfg}"
            # Add a slight drop shadow effect for text readability
            draw.text((x_offset + 12, y_offset + 12), label, fill=(0, 0, 0))
            draw.text((x_offset + 10, y_offset + 10), label, fill=(255, 255, 255))

    grid_path = os.path.join(args.output_dir, f"summary_grid_alpha{args.alpha:.1f}.png")
    grid_img.save(grid_path)
    print(f"[Success] Summary grid saved to: {grid_path}")