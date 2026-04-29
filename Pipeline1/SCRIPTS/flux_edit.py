import os

# 🔥 IMPORTANT: fix CUDA fragmentation BEFORE torch loads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import FluxFillPipeline


# -----------------------------
# Performance tweaks
# -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# -----------------------------
# Utilities
# -----------------------------
def resize_image(img, size=768):  # 🔥 reduced from 1024 → safer
    w, h = img.size
    scale = size / max(w, h)
    w, h = int(w * scale), int(h * scale)
    w, h = (w // 8) * 8, (h // 8) * 8
    return img.resize((w, h), Image.LANCZOS)


def create_soft_mask(image):
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    draw.rectangle(
        [w * 0.2, h * 0.2, w * 0.8, h * 0.8],
        fill=255
    )

    mask = mask.filter(ImageFilter.GaussianBlur(40))
    return mask


# -----------------------------
# Main
# -----------------------------
def main(input_img):
    model_id = "black-forest-labs/FLUX.1-Fill-dev"

    prompt = "Make the person look slimmer and more fit, realistic photo"
    output_path = os.path.splitext(input_img)[0] + "_edited.png"

    # -----------------------------
    # Load pipeline (STABLE SETUP)
    # -----------------------------
    pipe = FluxFillPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16  # 🔥 best for your GPU
    )

    pipe.to("cuda")

    # 🔥 Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()

    # -----------------------------
    # Load image
    # -----------------------------
    image = Image.open(input_img).convert("RGB")
    image = resize_image(image)

    # -----------------------------
    # Mask
    # -----------------------------
    mask = create_soft_mask(image)

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=22.0,        # 🔥 safer than 30
            num_inference_steps=22,     # 🔥 reduced for memory
            max_sequence_length=512,
        ).images[0]

    # -----------------------------
    # Save
    # -----------------------------
    result.save(output_path)

    print("\n✅ Done!")
    print(f"Saved at: {output_path}")


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    input_img = "/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS/slim.png"
    main(input_img)