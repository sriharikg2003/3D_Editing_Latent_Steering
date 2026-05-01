#!/usr/bin/env python3
"""
Render MP4s and 10 fixed-angle snapshots for each interpolated edit_mesh.glb.

For each alpha_X.XX/edit_mesh.glb found inside --interp_dir, this script:
  1. Renders a smooth 360° rotating MP4  (36 frames, 12 fps)
  2. Renders 10 snapshots at identical yaw angles (every 36°, same across
     all alpha levels so you can compare differences side-by-side)
  3. Saves a side-by-side comparison image for every snapshot angle

Usage:
    python3 render_interpolation.py --interp_dir ./output_interp
    python3 render_interpolation.py --interp_dir ./output_interp --engine EEVEE --resolution 512
"""

import os
import re
import sys
import math
import glob
import argparse
import numpy as np
import bpy
import imageio
from PIL import Image, ImageDraw, ImageFont
from mathutils import Vector
from pathlib import Path

# ─────────────────────────── camera constants ──────────────────────────────
N_SNAPSHOT_ANGLES = 10
N_VIDEO_FRAMES    = 36
SNAPSHOT_YAWS     = [i * (2 * math.pi / N_SNAPSHOT_ANGLES) for i in range(N_SNAPSHOT_ANGLES)]
PITCH             = 0.25          # slight downward tilt (radians)
RADIUS            = 2.0
FOV_DEG           = 40
FOV               = FOV_DEG / 180 * math.pi


# ───────────────────────── low-level render helpers ────────────────────────

def _setup_render(engine: str, resolution: int):
    bpy.context.scene.render.engine              = engine
    bpy.context.scene.render.resolution_x        = resolution
    bpy.context.scene.render.resolution_y        = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format  = "PNG"
    bpy.context.scene.render.image_settings.color_mode   = "RGBA"
    bpy.context.scene.render.film_transparent     = True

    if engine == "CYCLES":
        bpy.context.scene.cycles.samples         = 64
        bpy.context.scene.cycles.use_denoising   = True
        bpy.context.scene.cycles.filter_type     = "BOX"
        bpy.context.scene.cycles.filter_width    = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces  = 1
        try:
            bpy.context.scene.cycles.device = "GPU"
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        except Exception:
            pass
    else:  # BLENDER_EEVEE
        try:
            bpy.context.scene.eevee.taa_render_samples = 32
        except AttributeError:
            pass  # EEVEE Next (Blender 4.x) uses different settings


def _clear_scene():
    for col in [bpy.data.objects, bpy.data.materials,
                bpy.data.textures, bpy.data.images]:
        for item in list(col):
            col.remove(item, do_unlink=True)


def _load_glb(path: str):
    bpy.ops.import_scene.gltf(
        filepath=path, merge_vertices=True, import_shading="NORMALS"
    )


def _normalize_scene():
    """Scale and center all objects; returns (scale, offset)."""
    root_objs = [o for o in bpy.context.scene.objects if not o.parent]
    if len(root_objs) > 1:
        parent = bpy.data.objects.new("_root", None)
        bpy.context.scene.collection.objects.link(parent)
        for o in root_objs:
            o.parent = parent
    else:
        parent = root_objs[0]

    meshes = [o for o in bpy.context.scene.objects if isinstance(o.data, bpy.types.Mesh)]
    bbox_min = [math.inf] * 3
    bbox_max = [-math.inf] * 3
    for obj in meshes:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            bbox_min = [min(bbox_min[i], world[i]) for i in range(3)]
            bbox_max = [max(bbox_max[i], world[i]) for i in range(3)]

    scale  = 1.0 / max(b - a for a, b in zip(bbox_min, bbox_max))
    offset = Vector([-0.5 * (a + b) for a, b in zip(bbox_min, bbox_max)])

    parent.scale = parent.scale * scale
    bpy.context.view_layer.update()
    parent.matrix_world.translation += offset * scale
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def _add_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj  = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.data.sensor_height = cam_obj.data.sensor_width = 32

    # Always look at origin
    constraint              = cam_obj.constraints.new(type="TRACK_TO")
    constraint.track_axis   = "TRACK_NEGATIVE_Z"
    constraint.up_axis      = "UP_Y"
    empty                   = bpy.data.objects.new("_target", None)
    empty.location          = (0, 0, 0)
    bpy.context.scene.collection.objects.link(empty)
    constraint.target       = empty
    return cam_obj


def _add_lighting():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    for name, kind, energy, loc in [
        ("Key",  "POINT", 1200, (0, 2, 3)),
        ("Fill", "AREA",  8000, (0, 0, 8)),
        ("Back", "AREA",   800, (0, 0, -8)),
    ]:
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, type=kind))
        bpy.context.collection.objects.link(light)
        light.data.energy = energy
        light.location    = loc
        if kind == "AREA":
            light.scale = (50, 50, 50)


def _place_camera(cam, yaw: float, pitch: float = PITCH,
                  radius: float = RADIUS, fov: float = FOV):
    cam.location = (
        radius * math.cos(yaw) * math.cos(pitch),
        radius * math.sin(yaw) * math.cos(pitch),
        radius * math.sin(pitch),
    )
    cam.data.lens = 16 / math.tan(fov / 2)


def _render_to(path: str):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.context.view_layer.update()


# ───────────────────────────── main per-GLB function ───────────────────────

def render_glb(glb_path: str, output_dir: str, engine: str, resolution: int):
    """
    Render one GLB file:
      - 36-frame 360° rotating MP4  → output_dir/rotating.mp4
      - 10 fixed-angle snapshots    → output_dir/snapshot_00_yaw000.png …
    Returns list of snapshot paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "_tmp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    _setup_render(engine, resolution)
    _clear_scene()
    _load_glb(glb_path)
    _normalize_scene()
    cam = _add_camera()
    _add_lighting()

    # ── 1. Rotating video frames (0° → 360°) ──
    video_frame_paths = []
    yaws = np.linspace(0, 2 * math.pi, N_VIDEO_FRAMES, endpoint=False)
    print(f"  Rendering {N_VIDEO_FRAMES} video frames …", flush=True)
    for i, yaw in enumerate(yaws):
        _place_camera(cam, yaw)
        p = os.path.join(tmp_dir, f"vf_{i:03d}.png")
        _render_to(p)
        video_frame_paths.append(p)

    # Compile MP4
    frames_rgb = [np.array(Image.open(p).convert("RGB"))
                  for p in video_frame_paths if os.path.exists(p)]
    mp4_path = os.path.join(output_dir, "rotating.mp4")
    if frames_rgb:
        imageio.mimsave(mp4_path, frames_rgb, fps=12)
        print(f"  MP4 saved  → {mp4_path}")

    # Clean up temp frames
    for p in video_frame_paths:
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(tmp_dir)

    # ── 2. Fixed-angle snapshots ──
    snapshot_paths = []
    print(f"  Rendering {N_SNAPSHOT_ANGLES} snapshots …", flush=True)
    for i, yaw in enumerate(SNAPSHOT_YAWS):
        _place_camera(cam, yaw)
        deg = int(math.degrees(yaw))
        snap_path = os.path.join(output_dir, f"snapshot_{i:02d}_yaw{deg:03d}.png")
        _render_to(snap_path)
        snapshot_paths.append(snap_path)
        print(f"    snapshot {i+1}/{N_SNAPSHOT_ANGLES}  yaw={deg}°  → {snap_path}")

    return mp4_path, snapshot_paths


# ──────────────────── comparison image builder ─────────────────────────────

def _label_image(img: Image.Image, label: str, font_size: int = 18) -> Image.Image:
    """Add a text label at the bottom of an image."""
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    # Try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    w, h = out.size
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([(w//2 - tw//2 - 4, h - th - 8), (w//2 + tw//2 + 4, h - 2)], fill=(0, 0, 0, 200))
    draw.text((w//2 - tw//2, h - th - 6), label, fill=(255, 255, 255), font=font)
    return out


def build_comparison_images(
    all_snapshots: dict,   # {alpha_label: [path0, path1, …path9]}
    comparison_dir: str,
    thumb_size: int = 256
):
    """
    For each of the 10 snapshot angles, save a horizontal strip of
    all alpha levels so differences are immediately visible.
    all_snapshots keys are sorted alpha labels, e.g. "α=0.00", "α=0.20", …
    """
    os.makedirs(comparison_dir, exist_ok=True)
    n_alphas = len(all_snapshots)
    labels   = list(all_snapshots.keys())
    paths_per_alpha = list(all_snapshots.values())

    for angle_idx in range(N_SNAPSHOT_ANGLES):
        row_imgs = []
        for a_idx, label in enumerate(labels):
            snap_path = paths_per_alpha[a_idx][angle_idx]
            if os.path.exists(snap_path):
                img = Image.open(snap_path).convert("RGB").resize(
                    (thumb_size, thumb_size), Image.LANCZOS
                )
            else:
                img = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 30))
            img = _label_image(img, label)
            row_imgs.append(img)

        strip = Image.new("RGB", (thumb_size * n_alphas, thumb_size))
        for i, img in enumerate(row_imgs):
            strip.paste(img, (i * thumb_size, 0))

        deg = int(math.degrees(SNAPSHOT_YAWS[angle_idx]))
        out_path = os.path.join(comparison_dir, f"compare_angle_{angle_idx:02d}_yaw{deg:03d}.png")
        strip.save(out_path)
        print(f"  Comparison saved → {out_path}")


def build_comparison_mp4(
    all_snapshots: dict,
    comparison_dir: str,
    thumb_size: int = 256,
    fps: int = 2
):
    """
    MP4 that cycles through the 10 comparison strips (one frame per angle).
    """
    frames = []
    for angle_idx in range(N_SNAPSHOT_ANGLES):
        deg      = int(math.degrees(SNAPSHOT_YAWS[angle_idx]))
        img_path = os.path.join(comparison_dir, f"compare_angle_{angle_idx:02d}_yaw{deg:03d}.png")
        if os.path.exists(img_path):
            frames.append(np.array(Image.open(img_path).convert("RGB")))

    if frames:
        mp4_path = os.path.join(comparison_dir, "comparison_all_angles.mp4")
        imageio.mimsave(mp4_path, frames, fps=fps)
        print(f"  Comparison MP4 → {mp4_path}")


# ──────────────────────────────── main ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render rotating MP4s + fixed-angle snapshots for each interpolated GLB."
    )
    parser.add_argument(
        "--interp_dir", type=str, default="./output_interp",
        help="Directory produced by inference_interpolation.py (contains alpha_X.XX/ subdirs)"
    )
    parser.add_argument(
        "--engine", type=str, default="BLENDER_EEVEE", choices=["BLENDER_EEVEE", "CYCLES"],
        help="Blender render engine. BLENDER_EEVEE is ~10x faster; CYCLES is higher quality."
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Render resolution (square)"
    )
    parser.add_argument(
        "--thumb_size", type=int, default=256,
        help="Thumbnail size in comparison strip images"
    )
    args = parser.parse_args()

    interp_dir     = args.interp_dir
    comparison_dir = os.path.join(interp_dir, "comparison")

    # ── Find all alpha GLB files, sorted by alpha value ──
    pattern  = os.path.join(interp_dir, "alpha_*", "edit_mesh.glb")
    glb_files = sorted(glob.glob(pattern))

    if not glb_files:
        print(f"[ERROR] No edit_mesh.glb files found under: {interp_dir}")
        print("  Expected structure:  <interp_dir>/alpha_0.00/edit_mesh.glb  etc.")
        sys.exit(1)

    print(f"Found {len(glb_files)} interpolated meshes:")
    for p in glb_files:
        print(f"  {p}")

    # ── Render each GLB ──
    all_snapshots = {}   # alpha_label → [snap_path × 10]

    for glb_path in glb_files:
        # Extract alpha value from folder name
        alpha_folder = Path(glb_path).parent.name          # e.g. "alpha_0.40"
        match        = re.search(r"alpha_(\d+\.\d+)", alpha_folder)
        alpha_val    = float(match.group(1)) if match else 0.0
        alpha_label  = f"α={alpha_val:.2f}"

        render_dir = os.path.join(Path(glb_path).parent, "renders")
        print(f"\n{'='*60}")
        print(f"  Rendering  {alpha_label}  →  {render_dir}")
        print(f"{'='*60}")

        _, snapshot_paths = render_glb(
            glb_path   = glb_path,
            output_dir = render_dir,
            engine     = args.engine,
            resolution = args.resolution,
        )
        all_snapshots[alpha_label] = snapshot_paths

    # ── Build comparison strips + video ──
    print(f"\n{'='*60}")
    print(f"  Building comparison images and video")
    print(f"{'='*60}")
    build_comparison_images(all_snapshots, comparison_dir, thumb_size=args.thumb_size)
    build_comparison_mp4(all_snapshots, comparison_dir, thumb_size=args.thumb_size, fps=2)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Per-alpha renders  :  <interp_dir>/alpha_X.XX/renders/")
    print(f"    rotating.mp4       - smooth 360° video")
    print(f"    snapshot_XX_yawYYY.png - 10 fixed-angle frames")
    print(f"  Comparison strips  :  {comparison_dir}/")
    print(f"    compare_angle_XX_yawYYY.png  - all alphas side-by-side at one angle")
    print(f"    comparison_all_angles.mp4    - cycles through all 10 angles")
    print("=" * 60)
