import os
import gc
import torch
import numpy as np
from PIL import Image
import imageio

os.environ['SPCONV_ALGO'] = 'native'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

def get_latent_mean(pipeline, image_path):
    with torch.no_grad():
        print(f"Sampling latent for direction: {os.path.basename(image_path)}")
        img = Image.open(image_path)
        slat = pipeline.sample_slat_one_image(img)
        feat_mean = slat.feats.mean(dim=0).cpu()
        del slat
        torch.cuda.empty_cache()
        gc.collect()
        return feat_mean

def run_normalized_interpolation():
    model_path = "/mnt/data/srihari/MODELS/TRELLIS-image-large"
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()

    img_slim = '/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/slim.png'
    img_thick = '/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/thick.png'
    img_base = '/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/scale1.jpeg'

    feat_slim = get_latent_mean(pipeline, img_slim)
    feat_thick = get_latent_mean(pipeline, img_thick)
    direction = (feat_thick - feat_slim)
    del feat_slim, feat_thick

    print(f"Generating base SLAT for: {os.path.basename(img_base)}")
    with torch.no_grad():
        slat_base = pipeline.sample_slat_one_image(Image.open(img_base))

    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for alpha in alpha_values:
        print(f"\n--- Alpha: {alpha} (Normalized) ---")
        
        with torch.no_grad():
            # 1. Apply Direction
            dir_gpu = direction.to(slat_base.device)
            raw_edited_feats = slat_base.feats + (alpha * dir_gpu)
            
            # 2. Normalize to Original Norm
            # eps avoids division by zero
            eps = 1e-8
            orig_norm = torch.norm(slat_base.feats, p=2, dim=-1, keepdim=True)
            new_norm = torch.norm(raw_edited_feats, p=2, dim=-1, keepdim=True)
            
            # Re-scale features so magnitude matches the base object
            normalized_feats = raw_edited_feats * (orig_norm / (new_norm + eps))
            
            # 3. Replace and Decode
            slat_edited = slat_base.replace(normalized_feats)

            print(f"Decoding Gaussian...")
            out_gs = pipeline.decode_slat(slat_edited, formats=['gaussian'])
            video_gs = render_utils.render_video(out_gs['gaussian'][0])['color']
            imageio.mimsave(f"norm_alpha_{alpha}_gs.mp4", video_gs, fps=30)
            
            del out_gs, video_gs, slat_edited, normalized_feats, raw_edited_feats
            torch.cuda.empty_cache()
            gc.collect()

    print("\nNormalization pipeline complete.")

if __name__ == "__main__":
    run_normalized_interpolation()