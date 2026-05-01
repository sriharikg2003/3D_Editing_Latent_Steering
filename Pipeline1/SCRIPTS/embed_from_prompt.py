import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

import os
import torch
import numpy as np

# Import the pipeline using the correct module path from your directory structure
from trellis.pipelines.trellis_text_to_3d import TrellisTextTo3DPipeline

def run_clip_experiment():
    """
    Initializes the Trellis pipeline, extracts CLIP text embeddings, 
    and saves them for future analysis.
    """
    # 1. Initialize the pipeline
    # NOTE: Replace 'YOUR_MODEL_PATH' with the actual local path or HuggingFace repo ID
    model_path = "microsoft/TRELLIS-text-xlarge"
    print(f"Loading pipeline from: {model_path}...")
    
    try:
        pipeline = TrellisTextTo3DPipeline.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load the model. Ensure the path is correct. Error: {e}")
        return
    
    # 2. Define your experiment prompts
    prompts = [
        "change color to green"
    ]
    print(f"Extracting CLIP embeddings for {len(prompts)} prompts...")
    
    # 3. Extract embeddings and conditioning
    # Using get_cond() ensures we get both the positive text embedding and the negative (null) embedding
    with torch.no_grad():
        cond_dict = pipeline.get_cond(prompts)
        
    # 4. Move tensors to CPU for saving
    positive_embeddings = cond_dict['cond'].cpu()
    negative_embeddings = cond_dict['neg_cond'].cpu()
    
    # 5. Save the data
    # Create an output directory to keep the workspace clean
    save_dir = "experiment_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as a PyTorch dictionary (.pt) - highly recommended as it preserves structure
    torch_save_path = os.path.join(save_dir, 'clip_conditioning.pt')
    torch.save({
        'prompts': prompts,
        'positive_embeddings': positive_embeddings,
        'negative_embeddings': negative_embeddings
    }, torch_save_path)
    print(f"Saved PyTorch embeddings dictionary to {torch_save_path}")
    print(f"Positive tensor shape: {positive_embeddings.shape}")

    # Optional: Save just the positive embeddings as a NumPy array (.npy) if needed for other libraries
    numpy_save_path = os.path.join(save_dir, 'positive_embeddings.npy')
    np.save(numpy_save_path, positive_embeddings.numpy())
    print(f"Saved NumPy positive embeddings to {numpy_save_path}")

if __name__ == "__main__":
    # Ensure CUDA is available since the pipeline hardcodes .cuda() in some methods
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. The Trellis pipeline methods utilizing .cuda() will fail.")
    else:
        run_clip_experiment()
        print("Experiment completed successfully!")