import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


CONTROLNET_CONFIGS = {
    "canny": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "preprocessor": "canny",
    },
    "depth": {
        "model_id": "lllyasviel/sd-controlnet-depth",
        "preprocessor": "depth",
    },
    "pose": {
        "model_id": "lllyasviel/sd-controlnet-openpose",
        "preprocessor": "pose",
    },
    "hed": {
        "model_id": "lllyasviel/sd-controlnet-hed",
        "preprocessor": "hed",
    },
}

SD_BASE = "runwayml/stable-diffusion-v1-5"


def load_pipeline(control_type: str, device: str) -> tuple:
    cfg = CONTROLNET_CONFIGS[control_type]

    controlnet = ControlNetModel.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_BASE,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def preprocess_image(image_path: str, resolution: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = resolution / max(w, h)
    new_w = int((w * scale) // 8) * 8
    new_h = int((h * scale) // 8) * 8
    return img.resize((new_w, new_h), Image.LANCZOS)


def extract_control_map(
    image: Image.Image,
    control_type: str,
    canny_low: int,
    canny_high: int,
) -> Image.Image:
    if control_type == "canny":
        arr = np.array(image)
        edges = cv2.Canny(arr, canny_low, canny_high)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_rgb)

    elif control_type == "depth":
        detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        return detector(image)

    elif control_type == "pose":
        detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        return detector(image)

    elif control_type == "hed":
        detector = CannyDetector()
        return detector(image)

    else:
        raise ValueError(f"Unknown control_type: {control_type}")


def run_edit(
    pipe: StableDiffusionControlNetPipeline,
    control_map: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    num_images: int,
    seed: int,
) -> list:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_map,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images
    return results


def save_outputs(
    input_image: Image.Image,
    control_map: Image.Image,
    edited_images: list,
    output_dir: str,
    prompt: str,
    control_type: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_image.save(out / f"input_{ts}.png")
    control_map.save(out / f"control_map_{control_type}_{ts}.png")

    for i, img in enumerate(edited_images):
        img.save(out / f"edited_{ts}_{i:02d}.png")

    meta_path = out / f"meta_{ts}.txt"
    with open(meta_path, "w") as f:
        f.write(f"timestamp: {ts}\n")
        f.write(f"prompt: {prompt}\n")
        f.write(f"control_type: {control_type}\n")
        f.write(f"num_outputs: {len(edited_images)}\n")

    print(f"Saved {2 + len(edited_images)} files to {out.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Structure-preserving image editing via Stable Diffusion + ControlNet"
    )
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument(
        "--prompt", required=True,
        help="Text prompt for the output, e.g. 'a golden temple at sunset, detailed, 4k'"
    )
    p.add_argument(
        "--negative_prompt", default="blurry, low quality, deformed, watermark",
        help="Negative prompt"
    )
    p.add_argument(
        "--control_type", default="canny",
        choices=list(CONTROLNET_CONFIGS.keys()),
        help=(
            "canny  -> edge structure (objects/scenes)\n"
            "depth  -> depth map (3D structure)\n"
            "pose   -> human skeleton (people)\n"
            "hed    -> soft edges (artistic)"
        ),
    )
    p.add_argument("--output_dir", default="outputs", help="Directory to save results")
    p.add_argument("--resolution", type=int, default=512,
                   help="Max side resolution (512 or 768)")
    p.add_argument("--steps", type=int, default=30,
                   help="Number of diffusion steps (20-50)")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="Text guidance scale")
    p.add_argument("--controlnet_scale", type=float, default=1.0,
                   help="ControlNet conditioning strength (0.5-1.5)")
    p.add_argument("--canny_low", type=int, default=100,
                   help="Canny low threshold (only used with --control_type canny)")
    p.add_argument("--canny_high", type=int, default=200,
                   help="Canny high threshold (only used with --control_type canny)")
    p.add_argument("--num_images", type=int, default=1,
                   help="Number of output images to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(f"Device        : {args.device}")
    print(f"Image         : {args.image}")
    print(f"Prompt        : {args.prompt}")
    print(f"Control type  : {args.control_type}")

    image = preprocess_image(args.image, args.resolution)
    print(f"Input size after resize: {image.size}")

    print("Extracting control map ...")
    control_map = extract_control_map(image, args.control_type, args.canny_low, args.canny_high)

    print("Loading pipeline ...")
    pipe = load_pipeline(args.control_type, args.device)

    print("Running diffusion ...")
    edited = run_edit(
        pipe=pipe,
        control_map=control_map,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        num_images=args.num_images,
        seed=args.seed,
    )

    save_outputs(image, control_map, edited, args.output_dir, args.prompt, args.control_type)


if __name__ == "__main__":
    main()