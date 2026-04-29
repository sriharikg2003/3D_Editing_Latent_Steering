import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH      = "scale1.jpeg"
MASK_PATH       = None
OUT_DIR         = "outputs"
PROMPT          = "a realistic high quality detailed image"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

INFERENCE_STEPS = 100
JUMP_LENGTH     = 10
RESAMPLE_COUNT  = 10

# -----------------------------
# IMAGE / MASK UTILS
# -----------------------------
def load_image(path, size=512):
    return Image.open(path).convert("RGB").resize((size, size))


def make_box_mask(size=512, box=(150, 150, 350, 350)):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def load_or_create_mask(mask_path, size=512):
    if mask_path and os.path.exists(mask_path):
        m   = Image.open(mask_path).convert("L").resize((size, size))
        arr = (np.array(m) > 127).astype(np.uint8) * 255
        return Image.fromarray(arr)
    return make_box_mask(size)


def apply_mask_to_image(image, mask):
    img_arr  = np.array(image).astype(np.float32)
    mask_arr = np.array(mask)
    overlay  = img_arr.copy()
    overlay[mask_arr > 127] = [255, 0, 0]
    blended  = np.clip(overlay * 0.5 + img_arr * 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

# -----------------------------
# TENSOR HELPERS
# -----------------------------
def pil_to_tensor(img, device, dtype):
    x = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def pil_mask_to_tensor(mask, device, dtype):
    m = np.array(mask).astype(np.float32) / 255.0
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)


def tensor_to_pil(t):
    x = t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray(np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8))

# -----------------------------
# ENCODE / DECODE
# -----------------------------
def encode(vae, x):
    with torch.no_grad():
        return vae.encode(x).latent_dist.sample() * vae.config.scaling_factor


def decode(vae, z):
    with torch.no_grad():
        return vae.decode(z / vae.config.scaling_factor).sample

# -----------------------------
# NULL-TEXT EMBEDDING
# -----------------------------
def null_embedding(pipe, device, dtype):
    ids = pipe.tokenizer(
        "", return_tensors="pt", padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        return pipe.text_encoder(ids)[0].to(dtype)

# -----------------------------
# REPAINT CORE
#
# SD-inpainting UNet has conv_in weight shape [320, 9, 3, 3]:
#   channel layout = noisy_latent(4) | masked_image_latent(4) | mask(1)
#
# At each timestep t:
#   known   → re-noise x0 to level t  (forward q)
#   unknown → UNet reverse step        (p_theta)
# Resample r times per jump of j steps (RePaint §3.2)
# -----------------------------
def repaint_ddpm(pipe, image_pil, mask_pil, scheduler, n_steps, j, r):
    vae    = pipe.vae
    unet   = pipe.unet
    dtype  = next(unet.parameters()).dtype
    device = next(unet.parameters()).device

    image_t = pil_to_tensor(image_pil, device, dtype)           # (1,3,512,512)
    mask_t  = pil_mask_to_tensor(mask_pil, device, dtype)       # (1,1,512,512)  1=unknown

    z0          = encode(vae, image_t)                           # (1,4,64,64)
    mask_latent = torch.nn.functional.interpolate(
        mask_t, size=z0.shape[-2:], mode="nearest"
    )                                                            # (1,1,64,64)

    # masked-image latent: zero unknown pixels, encode
    mask_pixel      = torch.nn.functional.interpolate(
        mask_t, size=image_t.shape[-2:], mode="nearest"
    )
    masked_image_t  = image_t * (1 - mask_pixel)
    masked_z        = encode(vae, masked_image_t)                # (1,4,64,64)

    enc_hidden = null_embedding(pipe, device, dtype)             # (1,77,768)

    scheduler.set_timesteps(n_steps)
    timesteps = scheduler.timesteps                              # descending T…0

    x_t = torch.randn_like(z0)

    i = 0
    while i < len(timesteps):
        t = timesteps[i]

        for rep in range(r):
            unet_input = torch.cat([x_t, masked_z, mask_latent], dim=1)  # (1,9,64,64)

            with torch.no_grad():
                noise_pred = unet(
                    unet_input, t,
                    encoder_hidden_states=enc_hidden
                ).sample                                         # (1,4,64,64)

            x_prev = scheduler.step(noise_pred, t, x_t).prev_sample

            # paste known region at t_prev noise level
            if i + 1 < len(timesteps):
                t_prev      = timesteps[i + 1]
                noise       = torch.randn_like(z0)
                z0_at_tprev = scheduler.add_noise(z0, noise, t_prev.unsqueeze(0))
            else:
                z0_at_tprev = z0

            x_prev = (1 - mask_latent) * z0_at_tprev + mask_latent * x_prev

            # resample: re-noise back to t
            if rep < r - 1:
                noise = torch.randn_like(x_prev)
                x_t   = scheduler.add_noise(x_prev, noise, t.unsqueeze(0))
            else:
                x_t   = x_prev

        i += j

    return decode(vae, x_t)   # (1,3,512,512) in [-1,1]

# -----------------------------
# SD INPAINT BASELINE
# -----------------------------
def run_sd_inpaint(pipe, image, mask, prompt, n_steps):
    return pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=n_steps,
        guidance_scale=7.5,
    ).images[0]

# -----------------------------
# COMPOSITE: restore known pixels exactly
# -----------------------------
def composite(original_pil, inpainted_pil, mask_pil):
    orig    = np.array(original_pil)
    inp     = np.array(inpainted_pil)
    mask_np = np.array(mask_pil)
    out     = np.where(mask_np[:, :, None] > 127, inp, orig)
    return Image.fromarray(out.astype(np.uint8))

# -----------------------------
# PLOT
# -----------------------------
def plot_results(original, mask, masked_overlay, sd_result, repaint_result, out_path):
    images = [original, mask.convert("RGB"), masked_overlay, sd_result, repaint_result]
    titles = ["original", "mask\n(white=unknown)", "masked overlay",
              "SD inpaint\n(baseline)", "RePaint\n(resample)"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), facecolor="#0d0d0d")

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.array(img))
        ax.set_title(title, color="#cccccc", fontsize=8,
                     fontfamily="monospace", pad=5, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.text(0.5, 1.01, "mask inpainting: SD baseline vs RePaint resampling",
             ha="center", va="bottom", color="#eeeeee",
             fontsize=9, fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"plot → {out_path}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    image = load_image(IMAGE_PATH)
    mask  = load_or_create_mask(MASK_PATH)

    masked_overlay = apply_mask_to_image(image, mask)
    image.save(os.path.join(OUT_DIR, "original.png"))
    mask.save(os.path.join(OUT_DIR, "mask.png"))
    masked_overlay.save(os.path.join(OUT_DIR, "masked_overlay.png"))

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    print("loading pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
    ).to(DEVICE)

    print("SD inpainting baseline...")
    sd_result = run_sd_inpaint(pipe, image, mask, PROMPT, INFERENCE_STEPS)
    sd_result.save(os.path.join(OUT_DIR, "sd_inpaint.png"))

    print("RePaint resampling...")
    scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="scheduler"
    )

    repaint_raw  = repaint_ddpm(
        pipe, image, mask, scheduler,
        n_steps=INFERENCE_STEPS,
        j=JUMP_LENGTH,
        r=RESAMPLE_COUNT,
    )
    repaint_pil  = tensor_to_pil(repaint_raw)
    repaint_pil.save(os.path.join(OUT_DIR, "repaint_raw.png"))

    repaint_comp = composite(image, repaint_pil, mask)
    repaint_comp.save(os.path.join(OUT_DIR, "repaint_composite.png"))

    plot_results(image, mask, masked_overlay, sd_result, repaint_comp,
                 out_path=os.path.join(OUT_DIR, "comparison.png"))

    print("done. check outputs/")