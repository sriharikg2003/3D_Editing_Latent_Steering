import torch
import numpy as np
import os

def save_general_thickening_vector(thick_path, thin_path, save_name="general_thickening_direction.pt"):
    def load_stack(path):
        latents = []
        files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
        for f in files:
            # (1, 1373, 1024)
            latent = torch.load(os.path.join(path, f), map_location='cpu', weights_only=True)
            # Slice to get just the 1369 patches (skip CLS + 4 Regs)
            latents.append(latent[:, 5:, :])
        return torch.cat(latents, dim=0)

    # 1. Load and calculate raw difference
    X_thick = load_stack(thick_path)
    X_thin = load_stack(thin_path)
    
    # diff shape: (1369, 1024)
    diff = X_thick.mean(0) - X_thin.mean(0)
    
    # 2. Identify "Important" patches (the cylinder edges)
    patch_importance = torch.norm(diff, dim=-1)
    threshold = torch.quantile(patch_importance, 0.85) # Top 15% most active patches
    
    # 3. Compute the Semantic Mean Vector
    # We average only the vectors that showed significant change
    semantic_thickening_vector = diff[patch_importance > threshold].mean(0)
    
    # 4. Unit Normalization for steering stability
    semantic_thickening_vector = semantic_thickening_vector / (torch.norm(semantic_thickening_vector) + 1e-8)
    
    # 5. Save as a single (1024,) vector
    torch.save(semantic_thickening_vector, save_name)
    print(f"Saved General Direction Vector: {save_name}")
    print(f"Vector Shape: {semantic_thickening_vector.shape}")

# Paths
thick_dir = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS_LATENTS/thick_cylinders"
thin_dir = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS_LATENTS/thin_cylinders"

save_general_thickening_vector(thick_dir, thin_dir)