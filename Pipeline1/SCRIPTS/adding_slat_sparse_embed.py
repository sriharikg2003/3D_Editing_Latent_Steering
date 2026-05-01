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
    # saved_data_path = "experiment_results/clip_conditioning.pt"
    # if not os.path.exists(saved_data_path):
    #     print(f"Could not find {saved_data_path}. Please run the extraction script first.")
    #     return
        
    # saved_data = torch.load(saved_data_path)
    # base_embedding = saved_data['positive_embeddings'][0:1].to(device)

    color_prompt = "green color"

    with torch.no_grad():
        base_embedding_dict = pipeline.get_cond([color_prompt])
        base_embedding = base_embedding_dict['cond']
        neg_base_embedding = base_embedding_dict['neg_cond'] 

    
    # 3. Generate the new embedding (The "Bench")
    new_prompt = "A sleek wooden bench"
    print(f"Encoding new prompt: '{new_prompt}'")
    
    with torch.no_grad():
        new_cond_dict = pipeline.get_cond([new_prompt])
        new_embedding = new_cond_dict['cond']
        neg_cond = new_cond_dict['neg_cond'] 

    # 4. Setup the Output Directory
    output_dir = "alpha_blend_results_updated"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving all results to: {output_dir}/")

    norm_new_emb = torch.norm(new_cond_dict['cond'], dim = -1, keepdim = True)

    norm_base_emb = torch.norm(base_embedding, dim = -1, keepdim = True)

    normalized_base_embedding = base_embedding*(norm_new_emb/norm_base_emb)
    
    print("norm new emb ", norm_new_emb)

    print("norm_base_emb ", norm_base_emb)

    # 5. Define the Alpha values [0.1, 0.2, ..., 1.0]
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 6. The Experiment Loop
    for alpha in alphas:
        print(f"\n--- Generating for Alpha = {alpha:.1f} ---")
        
        # Calculate the blended embedding
        blended_embedding = (new_embedding + (alpha * normalized_base_embedding))
        norm1 = torch.norm(blended_embedding, dim = -1, keepdim = True)
        normalized_blended_embeding = blended_embedding * (norm_new_emb / norm1)
        print("norm1" , norm1)
        custom_cond = {
            'cond': normalized_blended_embeding,
            'neg_cond': neg_cond
        }

        # Reset seed inside the loop so the base noise is identical for every alpha!
        torch.manual_seed(42)
        num_samples = 1
        
        with torch.no_grad():
            coords = pipeline.sample_sparse_structure(
                cond=custom_cond, 
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

if __name__ == "__main__":
    alpha_blend_experiment()