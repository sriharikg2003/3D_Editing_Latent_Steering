import os
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

def worker(rank, num_gpus, num_samples, save_dir, model_path, image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ['SPCONV_ALGO'] = 'native'
    
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    
    img = Image.open(image_path)
    
    with torch.no_grad():
        image_cond = pipeline.preprocess_image(img)
        cond_dict = pipeline.encode_image(image_cond)
        base_latent = cond_dict['cond'] 

    samples_per_gpu = num_samples // num_gpus
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank != num_gpus - 1 else num_samples
    
    pbar = tqdm(total=end_idx - start_idx, desc=f"GPU {rank}", position=rank)
    
    for i in range(start_idx, end_idx):
        try:
            with torch.no_grad():
                noise = torch.randn_like(base_latent) * 0.15
                jittered_latent = base_latent + noise
                
                current_cond = {
                    'cond': jittered_latent,
                    'neg_cond': cond_dict['neg_cond']
                }
                
                coords = pipeline.sample_sparse_structure(
                    current_cond,
                    sampler_params={'steps': 20, 'cfg_strength': 7.5}
                )
                
                slat = pipeline.sample_slat(
                    current_cond,
                    coords,
                    sampler_params={'steps': 20, 'cfg_strength': 3.0}
                )
                
                outputs = pipeline.decode_slat(slat, formats=['gaussian'])
                gs = outputs['gaussian'][0]
                
                video = render_utils.render_video(gs)['color']
                imageio.mimsave(f"{save_dir}/videos/sample_{i}.mp4", video, fps=30)
                
                np.save(f"{save_dir}/latents/lat_{i}.npy", jittered_latent.cpu().numpy())

                del coords, slat, outputs, gs, video
                torch.cuda.empty_cache()
                pbar.update(1)
                
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    NUM_GPUS = 3
    NUM_SAMPLES = 500
    SAVE_DIR = "interfacegan_trellis"
    MODEL_PATH = "/mnt/data/srihari/MODELS/TRELLIS-image-large"
    IMAGE_PATH = "/mnt/data/srihari/my_TRELLIS/chair2.webp"
    
    os.makedirs(f"{SAVE_DIR}/latents", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/videos", exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(rank, NUM_GPUS, NUM_SAMPLES, SAVE_DIR, MODEL_PATH, IMAGE_PATH))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()