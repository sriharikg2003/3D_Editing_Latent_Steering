"""
create_mask.py — Text-guided 3D voxel mask generation.

Pipeline
--------
1. Grounding DINO   : Detect bounding boxes for the text prompt, selecting the SMALLEST valid box.
2. SAM              : Segment the bounding box into a 2D binary mask on the render view.
3. Ray casting      : Project each masked pixel into the voxel grid (stopping at first hit).
4. 3D dilation      : Expand the surface mask outward.
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

from inference.model_utils import VoxelProcessor

# ---------------------------------------------------------------------------
# Lazy Model Loaders
# ---------------------------------------------------------------------------

def _load_gdino(device: str):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print("[create_mask] Loading Grounding DINO ...")
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device).eval()
    return processor, model

def _load_sam(device: str):
    from transformers import SamModel, SamProcessor
    print("[create_mask] Loading SAM ...")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    return model, processor

# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------

def detect_bounding_box(image: Image.Image, text_prompt: str, processor, model, threshold: float) -> list | None:
    """Uses Grounding DINO to find the most specific (smallest) valid bounding box."""
    prompt = text_prompt if text_prompt.endswith(".") else text_prompt + "."
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # DINO filters out anything below the box_threshold automatically
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"].cpu(),
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]], 
    )[0]

    if len(results["scores"]) == 0:
        return None

    # BUG FIX: Instead of blindly taking the highest score (which is often the whole object),
    # calculate the area of all valid boxes and take the smallest one to ensure specificity.
    boxes = results["boxes"]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = areas.argmin().item()
   
    return boxes[best_idx].cpu().numpy().tolist()

def segment_image(image: Image.Image, bbox: list, model, processor) -> np.ndarray:
    """Uses SAM to create a 2D binary mask from the detected bounding box."""
    inputs = processor(
        images=image,
        input_boxes=[[[bbox[0], bbox[1], bbox[2], bbox[3]]]],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]

    # Pick the largest mask candidate — the bounding box is already
    # constrained to the target region, so the fullest coverage is best.
    areas = masks[0].float().sum(dim=(-1, -2))
    best_idx = areas.argmax().item()
    return masks[0, best_idx].numpy().astype(np.uint8)

def project_2d_to_3d_surface(
    mask_2d: np.ndarray,
    transform_matrix: list,
    fov: float,
    render_res: int,
    occupancy: np.ndarray,
    grid_size: int = 64,
    ray_steps: int = 256,
) -> np.ndarray:
    """Casts rays to project the 2D mask onto the 3D surface grid."""
    g = grid_size
    mask3d = np.zeros((g, g, g), dtype=np.uint8)

    c2w = np.array(transform_matrix, dtype=np.float64)
    c2w[:3, 1:3] *= -1 
    R = c2w[:3, :3]                
    cam_pos = c2w[:3, 3]                 

    f = (render_res / 2.0) / np.tan(fov / 2.0)
    cx = cy = render_res / 2.0

    h_m, w_m = mask_2d.shape
    ys, xs = np.where(mask_2d > 0)
    
    if len(xs) == 0:
        return mask3d

    px = (xs * render_res / w_m - cx) / f
    py = (ys * render_res / h_m - cy) / f
    pz = np.ones_like(px)

    dirs_cam = np.stack([px, py, pz], axis=1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    dirs_world = dirs_cam @ R.T

    # Track active rays to stop at the first surface hit
    active_rays = np.ones(len(xs), dtype=bool)
    # Compute the farthest corner of the [-0.5, 0.5]^3 voxel bounding box from
    # the camera. This guarantees every ray that enters the grid will fully exit
    # before we stop marching — regardless of camera distance or angle.
    corners = np.array([[sx, sy, sz]
                        for sx in (-0.5, 0.5)
                        for sy in (-0.5, 0.5)
                        for sz in (-0.5, 0.5)], dtype=np.float64)
    max_t = float(np.linalg.norm(corners - cam_pos, axis=1).max())
    # Step size must be ≤ half a voxel (1/grid_size) to avoid skipping occupied voxels.
    n_steps = max(ray_steps, int(max_t * grid_size * 2))
    t_vals = np.linspace(0.0, max_t, n_steps)

    for t in t_vals:
        if not np.any(active_rays):
            break

        pts = cam_pos + dirs_world * t
        vi = np.floor((pts + 0.5) * g).astype(np.int32)
        in_grid = np.all((vi >= 0) & (vi < g), axis=1)

        current_active = in_grid & active_rays
        if not np.any(current_active):
            continue

        active_indices = np.where(current_active)[0]
        vx = vi[active_indices, 0]
        vy = vi[active_indices, 1]
        vz = vi[active_indices, 2]

        hits = occupancy[vx, vy, vz].astype(bool)
        
        if np.any(hits):
            hit_ray_indices = active_indices[hits]
            mask3d[vi[hit_ray_indices, 0], vi[hit_ray_indices, 1], vi[hit_ray_indices, 2]] = 1
            active_rays[hit_ray_indices] = False # Deactivate rays that hit the surface

    return mask3d

def dilate_3d(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Applies spherical morphological dilation to the 3D mask."""
    if radius <= 0:
        return mask.copy()

    ki, kj, kk = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
    kernel = (ki**2 + kj**2 + kk**2 <= radius**2)
    return binary_dilation(mask.astype(bool), structure=kernel).astype(np.uint8)

# ---------------------------------------------------------------------------
# Main Public API
# ---------------------------------------------------------------------------

def create_mask_3d(
    image_path: str,
    text_prompt: str,
    render_dir: str,
    voxel_ply_path: str,
    output_dir: str | None = None,
    front_image_name: str = "front.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    score_threshold: float = 0.2, # Bumped slightly for cleaner valid boxes
    grid_size: int = 64,
    dilation_voxels: int = 4,
    dilation_pixels: int = 0,
) -> dict:
    """
    Generate a 3D voxel mask. 
    Strictly requires an input image and a text prompt.
    """
    if not text_prompt or not text_prompt.strip():
        raise ValueError("A valid 'text_prompt' is strictly required.")
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError("A valid 'image_path' is strictly required.")

    output_dir = output_dir or render_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Rendering Metadata & Images
    meta_path = os.path.join(render_dir, "front_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    transform_matrix = meta["transform_matrix"]
    fov = float(meta["camera_angle_x"])
    render_res = int(meta.get("resolution", 512))

    source_image = Image.open(image_path).convert("RGB")
    front_path = os.path.join(render_dir, front_image_name)
    front_image = Image.open(front_path).convert("RGB")

    # 2. Load Models
    gdino_proc, gdino_model = _load_gdino(device)
    sam_model, sam_proc = _load_sam(device)

    # 3. Detect on Source Image
    bbox_source = detect_bounding_box(source_image, text_prompt, gdino_proc, gdino_model, score_threshold)
    
    mask_2d_path = os.path.join(output_dir, "mask_2d.png")
    mask_ply_path = os.path.join(output_dir, "mask.ply")
    voxel_processor = VoxelProcessor(grid_size)

    if bbox_source is None:
        print(f"[create_mask] Nothing found for '{text_prompt}'.")
        empty_3d = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        voxel_processor.voxel_to_ply(empty_3d, mask_ply_path)
        Image.fromarray(np.zeros((render_res, render_res), dtype=np.uint8)).save(mask_2d_path)
        return {"mask_ply": mask_ply_path, "mask_2d_png": mask_2d_path, "n_masked": 0}

    # 4. Rescale BBox to match the Rendered 3D Camera view
    sx = front_image.width / source_image.width
    sy = front_image.height / source_image.height
    bbox_front = [
        bbox_source[0] * sx,
        bbox_source[1] * sy,
        bbox_source[2] * sx,
        bbox_source[3] * sy,
    ]

    # 5. Segment on the Rendered View (Critical for 3D Math Alignment)
    mask_2d = segment_image(front_image, bbox_front, sam_model, sam_proc)

    if dilation_pixels > 0:
        ki, kj = np.ogrid[-dilation_pixels:dilation_pixels + 1, -dilation_pixels:dilation_pixels + 1]
        kernel_2d = (ki**2 + kj**2 <= dilation_pixels**2)
        mask_2d = binary_dilation(mask_2d.astype(bool), structure=kernel_2d).astype(np.uint8)

    Image.fromarray((mask_2d * 255).astype(np.uint8)).save(mask_2d_path)
    print(f"[create_mask] 2D Mask saved. Foreground pixels: {mask_2d.sum()}")

    # 6. Project to 3D & Dilate
    occupancy_grid = voxel_processor.ply_to_voxel(voxel_ply_path)
    mask_3d = project_2d_to_3d_surface(
        mask_2d, transform_matrix, fov, render_res, occupancy_grid, grid_size
    )

    mask_3d = dilate_3d(mask_3d, dilation_voxels)
    voxel_processor.voxel_to_ply(mask_3d, mask_ply_path)
    print(f"[create_mask] 3D Mask saved with {int(mask_3d.sum())} active voxels.")

    return {
        "mask_ply": mask_ply_path,
        "mask_2d_png": mask_2d_path,
        "n_masked": int(mask_3d.sum()),
    }