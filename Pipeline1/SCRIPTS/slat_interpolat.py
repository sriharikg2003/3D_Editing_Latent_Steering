import os
import torch
import imageio
from trellis.pipelines.trellis_text_to_3d import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

def alpha_blend_experiment():
    # 1. Load the Pipeline (Once)
    model_path = "microsoft/TRELLIS-text-xlarge"
    print("Loading pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load the base embedding (The "Car" or whatever is saved)
    saved_data_path = "experiment_results/clip_conditioning.pt"
    if not os.path.exists(saved_data_path):
        print(f"Could not find {saved_data_path}. Please run the extraction script first.")
        return
        
   
    
    # 3. Generate the new embedding (The "Bench")
    new_prompt = "A sleek wooden bench"
    print(f"Encoding new prompt: '{new_prompt}'")

    with torch.no_grad():
        new_cond_dict = pipeline.get_cond([new_prompt])
        new_embedding = new_cond_dict['cond']
        neg_cond = new_cond_dict['neg_cond'] 



    ###prompts for texture


    prompt1 = [
        "change color to green"
    ]
    print(f"Extracting CLIP embeddings for {len(prompt1)} prompts...")
    
    # 3. Extract embeddings and conditioning
    # Using get_cond() ensures we get both the positive text embedding and the negative (null) embedding
    with torch.no_grad():
        cond_dict1 = pipeline.get_cond(prompt1)
        
    # 4. Move tensors to CPU for saving
    positive_embeddings1 = cond_dict1['cond']
    negative_embeddings1 = cond_dict1['neg_cond']


    prompt2 = [
        "change color to purple"
    ]
    print(f"Extracting CLIP embeddings for {len(prompt2)} prompts...")
    
    # 3. Extract embeddings and conditioning
    # Using get_cond() ensures we get both the positive text embedding and the negative (null) embedding
    with torch.no_grad():
        cond_dict2 = pipeline.get_cond(prompt2)
        
    # 4. Move tensors to CPU for saving
    positive_embeddings2 = cond_dict2['cond']
    negative_embeddings2 = cond_dict2['neg_cond']





    # 4. Setup the Output Directory
    output_dir = "alpha_blend_results_texture_interpol"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving all results to: {output_dir}/")

    # 5. Define the Alpha values [0.1, 0.2, ..., 1.0]
    alphas = [0,  0.2,  0.4, 0.5, 0.7,  0.9, 1.0]

    # 6. The Experiment Loop
    for alpha in alphas:
        print(f"\n--- Generating for Alpha = {alpha:.1f} ---")
        
        # Calculate the blended embedding
        blended_embedding = ((1-alpha) * positive_embeddings1 )+ (alpha *  positive_embeddings2 )
        
        custom_cond = {
            'cond': blended_embedding,
            'neg_cond': neg_cond
        }

        # Reset seed inside the loop so the base noise is identical for every alpha!
        torch.manual_seed(42)
        num_samples = 1
        
        with torch.no_grad():
            coords = pipeline.sample_sparse_structure(
                cond= new_cond_dict, 
                num_samples=num_samples, 
                sampler_params={}
            )
            
            slat = pipeline.sample_slat(
                cond=custom_cond, 
                coords=coords, 
                sampler_params={}
            )
            
            formats = ['mesh', 'gaussian', 'radiance_field']
            outputs = pipeline.decode_slat(slat, formats)
            
        print("Saving files...")
        
        # Save Videos with alpha in the filename
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(os.path.join(output_dir, f"alpha_{alpha:.1f}_gs.mp4"), video_gs, fps=30)
        
        video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
        imageio.mimsave(os.path.join(output_dir, f"alpha_{alpha:.1f}_rf.mp4"), video_rf, fps=30)
        
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(os.path.join(output_dir, f"alpha_{alpha:.1f}_mesh.mp4"), video_mesh, fps=30)

        # Save GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb.export(os.path.join(output_dir, f"alpha_{alpha:.1f}.glb"))

        # Save PLY
        outputs['gaussian'][0].save_ply(os.path.join(output_dir, f"alpha_{alpha:.1f}.ply"))

    print("\nExperiment complete! Check the 'alpha_blend_results' folder.")


def create_image_grid():
    output_dir = "alpha_blend_results_texture_interpol"
    alphas = [0.0, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0]
    formats = ['gs', 'rf', 'mesh']
    
    all_rows = []
    
    print("Extracting first frames and building grid...")
    
    for alpha in alphas:
        row_frames = []
        for fmt in formats:
            vid_path = os.path.join(output_dir, f"alpha_{alpha:.1f}_{fmt}.mp4")
            try:
                # 1. Open the video and grab the very first frame
                reader = imageio.get_reader(vid_path)
                first_frame = reader.get_data(0)
                reader.close()
                
                # 2. Convert to PIL Image to standardize size (prevents array mismatch errors)
                img = Image.fromarray(first_frame).resize((256, 256))
                row_frames.append(np.array(img))
                
            except Exception as e:
                print(f"Warning: Could not read {vid_path}: {e}")
                # Create a blank black square if a video is missing
                blank = np.zeros((256, 256, 3), dtype=np.uint8)
                row_frames.append(blank)
                
        # 3. Stitch the 3 formats (GS, RF, Mesh) horizontally for this alpha step
        row_image = np.concatenate(row_frames, axis=1)
        all_rows.append(row_image)
        
    # 4. Stitch all the alpha rows vertically into one massive grid
    final_grid_np = np.concatenate(all_rows, axis=0)
    final_image = Image.fromarray(final_grid_np)
    
    # 5. Save the output
    save_path = "experiment_summary_grid.png"
    final_image.save(save_path)
    
    print(f"\n✅ Success! Saved unified grid to: {save_path}")


if __name__ == "__main__":
    alpha_blend_experiment()

    create_image_grid()