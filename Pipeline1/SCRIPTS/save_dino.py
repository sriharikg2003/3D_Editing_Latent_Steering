import os
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
import sys

os.environ['SPCONV_ALGO'] = 'native'

pipeline = TrellisImageTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-image-large")
pipeline.cuda()

base_input_path = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS"
base_output_path = "/mnt/data/srihari/my_TRELLIS/CYLINDER_GENERATIONS_LATENTS"

if not os.path.exists(base_input_path):
    print(f"Error: Input path {base_input_path} does not exist.")
    sys.exit()

all_folders = [f for f in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, f))]

for folder in all_folders:
    input_dir = os.path.join(base_input_path, folder)
    output_dir = os.path.join(base_output_path, folder)
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.avif'))]
    
    if not image_files:
        continue
        
    print(f"Checking folder: {folder}")
    
    for img_name in image_files:
        output_filename = os.path.splitext(img_name)[0] + ".pt"
        save_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(save_path):
            continue

        img_path = os.path.join(input_dir, img_name)
        try:
            image = Image.open(img_path)
            processed_image = pipeline.preprocess_image(image)
            
            with torch.no_grad():
                cond_dict = pipeline.get_cond([processed_image])
                dino_features = cond_dict['cond']
                
            torch.save(dino_features.cpu(), save_path)
            print(f"Generated: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

print("Incremental extraction complete. Exiting.")
sys.exit()