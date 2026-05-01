import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import matplotlib.pyplot as plt

def generate_mask_from_prompt(image_path, text_prompt):
    # 1. Load the processor and model from Hugging Face
    print("Loading model...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # 2. Load and prepare the image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        return

    # 3. Process the image and the prompt
    print(f"Generating mask for prompt: '{text_prompt}'...")
    inputs = processor(
        text=[text_prompt], 
        images=[image], 
        padding="max_length", 
        return_tensors="pt"
    )

    # 4. Perform the prediction
    # 4. Perform the prediction
    with torch.no_grad():
        # Tell the model to explicitly handle the 352x352 resolution
        outputs = model(**inputs, interpolate_pos_encoding=True)
        
        # The output logits represent the raw, unnormalized mask predictions
        # We add an unsqueeze to match expected dimensions
        preds = outputs.logits.unsqueeze(1)

    mask = torch.sigmoid(preds[0][0])

    # --- SAVE OUTPUT 1: Side-by-side comparison plot ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap='viridis')
    ax[1].set_title(f'Mask for: "{text_prompt}"')
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    plt.close(fig) # Free up memory
    print("-> Saved comparison plot to: 'comparison_plot.png'")

    # --- SAVE OUTPUT 2: Colored Heatmap ---
    plt.imsave("heatmap_mask.png", mask.cpu().numpy(), cmap='viridis')
    print("-> Saved colored heatmap to: 'heatmap_mask.png'")

    # --- SAVE OUTPUT 3: Strict Black & White Mask ---
    binary_mask = (mask > 0.5).float()
    plt.imsave("binary_mask.png", binary_mask.cpu().numpy(), cmap='gray')
    print("-> Saved black & white mask to: 'binary_mask.png'")
    
    print("\nDone! Check your folder for the saved images.")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the path to an image on your computer
    IMAGE_PATH = "/data/home/divya1/projects/assign/Nano3D/images/lamp.jpeg"
    
    # Replace with what you want to mask (e.g., "a car", "the blue mug", "a dog")
    PROMPT = "Quadraple the size of the bulb downward only" 
    
    generate_mask_from_prompt(IMAGE_PATH, PROMPT)