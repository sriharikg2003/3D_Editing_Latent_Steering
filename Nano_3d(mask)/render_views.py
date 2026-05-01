#!/usr/bin/env python3
"""
render_views.py — Render front, back, left, and right views of one or more GLB files.

Usage (single model):
    python3 render_views.py --glb_paths model1.glb model2.glb --output_dir ./views
    python3 render_views.py --glb_paths ./output/src_mesh.glb --output_dir ./views --resolution 512

Usage (consolidated — one image per view across all alpha values):
    python3 render_views.py --mode consolidated \\
        --alpha_base_dir ./outputs --output_dir ./views/per_alpha \\
        --consolidated_dir ./views/consolidated
    Expects subdirs named alpha_0.0/, alpha_0.2/, ... each containing edit_mesh.glb.
"""

import os
import math
import argparse
import json
import shutil
import numpy as np
import bpy
from mathutils import Vector
from typing import Tuple, Dict, Optional


# Camera positions for each view (yaw in radians, pitch = 0)
VIEWS = {
    "front": -math.pi / 2,   # camera at (0, -r, 0) — looks along +Y
    "back":   math.pi / 2,   # camera at (0, +r, 0) — looks along -Y
    "right":  0.0,            # camera at (+r, 0, 0) — looks along -X
    "left":   math.pi,        # camera at (-r, 0, 0) — looks along +X
}


def _discover_alpha_glb_map(alpha_base_dir: str) -> Dict[float, str]:
    """Scan alpha_base_dir for subdirs named alpha_X.X/ containing edit_mesh.glb."""
    result = {}
    for entry in sorted(os.listdir(alpha_base_dir)):
        if not entry.startswith("alpha_"):
            continue
        try:
            alpha_val = float(entry[len("alpha_"):])
        except ValueError:
            continue
        glb = os.path.join(alpha_base_dir, entry, "edit_mesh.glb")
        if os.path.isfile(glb):
            result[alpha_val] = glb
    if not result:
        raise FileNotFoundError(
            f"No alpha_X.X/edit_mesh.glb found under: {alpha_base_dir}"
        )
    return result


def stitch_consolidated_views(
    per_alpha_views: Dict[float, Dict[str, str]],
    consolidated_dir: str,
    resolution: int = 512,
) -> Dict[str, str]:
    """
    Stitch per-alpha renders into 4 consolidated images, one per view direction.

    Layout of each consolidated image (e.g. consolidated_front.png):
        [ α=0.0 | α=0.2 | α=0.4 | … ]   ← resolution × resolution cells
        Label bar above each cell shows the alpha value.

    Args:
        per_alpha_views: {alpha_value: {view_name: image_path}}
        consolidated_dir: directory where consolidated_*.png are saved
        resolution: pixel size of each individual render cell

    Returns:
        {view_name: path_to_consolidated_image}
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "Pillow is required for stitching. Install it with: pip install Pillow"
        )

    os.makedirs(consolidated_dir, exist_ok=True)

    alphas = sorted(per_alpha_views.keys())
    n = len(alphas)
    label_h = 48                              # pixels reserved for the alpha label
    cell_w = resolution
    cell_h = resolution
    canvas_w = n * cell_w
    canvas_h = label_h + cell_h

    # Try to load a reasonably sized font; fall back to the PIL default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    saved: Dict[str, str] = {}
    for view_name in VIEWS:
        canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)

        for col, alpha in enumerate(alphas):
            x0 = col * cell_w

            # ── label ──────────────────────────────────────────────────────
            label = f"α = {alpha:.1f}"
            # Draw a white label band
            draw.rectangle([x0, 0, x0 + cell_w - 1, label_h - 1], fill=(255, 255, 255))
            # Vertical separator between columns
            if col > 0:
                draw.line([(x0, 0), (x0, canvas_h - 1)], fill=(180, 180, 180), width=2)

            # Centre the text in the label band
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text(
                (x0 + (cell_w - text_w) // 2, (label_h - text_h) // 2),
                label,
                fill=(30, 30, 30),
                font=font,
            )

            # ── render cell ────────────────────────────────────────────────
            img_path = per_alpha_views.get(alpha, {}).get(view_name)
            if img_path and os.path.isfile(img_path):
                img = Image.open(img_path).convert("RGBA")
                if img.size != (cell_w, cell_h):
                    img = img.resize((cell_w, cell_h), Image.LANCZOS)
                # Composite transparent render onto white background
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                cell = Image.alpha_composite(bg, img).convert("RGB")
                canvas.paste(cell, (x0, label_h))
            else:
                # Gray placeholder for missing renders
                draw.rectangle(
                    [x0, label_h, x0 + cell_w - 1, label_h + cell_h - 1],
                    fill=(200, 200, 200),
                )
                draw.text(
                    (x0 + cell_w // 2, label_h + cell_h // 2),
                    "missing",
                    fill=(120, 120, 120),
                    font=font,
                    anchor="mm",
                )

        out_path = os.path.join(consolidated_dir, f"consolidated_{view_name}.png")
        canvas.save(out_path)
        saved[view_name] = out_path
        print(f"[render_views] consolidated {view_name:5s} → {out_path}")

    return saved


class GLBViewRenderer:
    def __init__(self, resolution: int = 512, engine: str = "CYCLES"):
        self.resolution = resolution
        self.engine = engine
        # Standalone bpy (pip) has no window manager initialised by default.
        # read_factory_settings boots the WM and enables core addons.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _clear_scene(self):
        # Reset to a clean empty scene and re-enable the GLTF importer.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")

    def _init_render_settings(self):
        scene = bpy.context.scene
        scene.render.engine = self.engine
        scene.render.resolution_x = self.resolution
        scene.render.resolution_y = self.resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
        if self.engine == "CYCLES":
            scene.cycles.samples = 128
            scene.cycles.filter_type = "BOX"
            scene.cycles.filter_width = 1
            scene.cycles.diffuse_bounces = 1
            scene.cycles.glossy_bounces = 1
            scene.cycles.transparent_max_bounces = 3
            scene.cycles.transmission_bounces = 3
            scene.cycles.use_denoising = True
            try:
                scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            except Exception:
                pass

    def _load_glb(self, path: str):
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True, import_shading="NORMALS")

    def _normalize_scene(self) -> Tuple[float, Vector]:
        """Scale and center the scene so it fits in a unit cube."""
        root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(root_objects) > 1:
            parent = bpy.data.objects.new("SceneRoot", None)
            bpy.context.scene.collection.objects.link(parent)
            for obj in root_objects:
                obj.parent = parent
            root = parent
        else:
            root = root_objects[0]

        bbox_min, bbox_max = self._scene_bbox()
        scale = 1.0 / max(bbox_max - bbox_min)
        root.scale *= scale
        bpy.context.view_layer.update()

        bbox_min, bbox_max = self._scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        root.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return scale, offset

    def _scene_bbox(self) -> Tuple[Vector, Vector]:
        inf = math.inf
        bbox_min = [inf, inf, inf]
        bbox_max = [-inf, -inf, -inf]
        meshes = [obj for obj in bpy.context.scene.objects.values()
                  if isinstance(obj.data, bpy.types.Mesh)]
        if not meshes:
            raise RuntimeError("No mesh objects found in scene.")
        for obj in meshes:
            for corner in obj.bound_box:
                world = obj.matrix_world @ Vector(corner)
                for i in range(3):
                    bbox_min[i] = min(bbox_min[i], world[i])
                    bbox_max[i] = max(bbox_max[i], world[i])
        return Vector(bbox_min), Vector(bbox_max)

    def _init_camera(self):
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32

        # Always track to origin
        empty = bpy.data.objects.new("CameraTarget", None)
        empty.location = (0, 0, 0)
        bpy.context.scene.collection.objects.link(empty)
        constraint = cam.constraints.new(type="TRACK_TO")
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"
        constraint.target = empty
        return cam

    def _init_lighting(self):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        def add_light(name, light_type, energy, location):
            data = bpy.data.lights.new(name, type=light_type)
            obj = bpy.data.objects.new(name, data)
            bpy.context.collection.objects.link(obj)
            obj.data.energy = energy
            obj.location = location
            return obj

        add_light("KeyLight",  "POINT", 1500, (0,  2,  3))
        top = add_light("TopLight",  "AREA",  8000, (0,  0,  8))
        top.scale = (50, 50, 50)
        add_light("FillLight", "AREA",   800, (0,  0, -8))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_four_views(
        self,
        glb_path: str,
        output_dir: str,
        front_white_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Render front, back, left, and right views of a GLB file.

        Args:
            glb_path:         Absolute or relative path to the .glb file.
            output_dir:       Directory where the four PNG images are saved.
            front_white_path: Optional path to an already-processed front image
                              (e.g. output/image/front_white.png from bg_to_white).
                              When provided the front view is not re-rendered from
                              Blender — the image is copied directly.

        Returns:
            Dict mapping view name → saved image path.
        """
        os.makedirs(output_dir, exist_ok=True)

        self._clear_scene()
        self._init_render_settings()
        self._load_glb(os.path.abspath(glb_path))
        self._normalize_scene()

        cam = self._init_camera()
        self._init_lighting()

        radius = 2.0
        fov    = 40.0 / 180.0 * math.pi          # 40° vertical FOV
        cam.data.lens = 16.0 / math.tan(fov / 2)  # sensor_width=32 → lens = 16/tan(fov/2)

        saved = {}
        stem = os.path.splitext(os.path.basename(glb_path))[0]

        for view_name, yaw in VIEWS.items():
            out_path = os.path.join(output_dir, f"{stem}_{view_name}.png")

            # Use the supplied white-background front image instead of rendering
            if view_name == "front" and front_white_path is not None:
                shutil.copy2(front_white_path, out_path)
                saved[view_name] = out_path
                print(f"[render_views] {'front':5s} → {out_path}  (copied from {front_white_path})")
                continue

            cam.location = (
                radius * math.cos(yaw),   # X
                radius * math.sin(yaw),   # Y
                0.0,                      # Z (pitch = 0)
            )
            bpy.context.view_layer.update()

            bpy.context.scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)

            saved[view_name] = out_path
            print(f"[render_views] {view_name:5s} → {out_path}")

        # Save a small metadata sidecar
        meta_path = os.path.join(output_dir, f"{stem}_views.json")
        with open(meta_path, "w") as f:
            json.dump({
                "glb": glb_path,
                "fov_degrees": 40.0,
                "radius": radius,
                "views": {k: {"yaw_deg": round(math.degrees(v), 1), "path": saved[k]}
                          for k, v in VIEWS.items()},
            }, f, indent=4)

        return saved

    def render_all_alphas_consolidated(
        self,
        alpha_glb_map: Dict[float, str],
        output_dir: str,
        consolidated_dir: str,
    ) -> Dict[str, str]:
        """
        Render four views for every alpha in alpha_glb_map, then stitch each
        view direction into one consolidated image.

        Args:
            alpha_glb_map:    {alpha_value: path_to_glb}
            output_dir:       directory for per-alpha individual renders
                              (subdirs alpha_X.X_views/ will be created here)
            consolidated_dir: directory for the 4 consolidated images

        Returns:
            {view_name: path_to_consolidated_image}
        """
        per_alpha_views: Dict[float, Dict[str, str]] = {}

        for alpha, glb_path in sorted(alpha_glb_map.items()):
            if not os.path.isfile(glb_path):
                print(f"[render_views] WARNING: GLB not found for α={alpha:.1f}, skipping: {glb_path}")
                continue
            alpha_out = os.path.join(output_dir, f"alpha_{alpha:.1f}_views")
            print(f"\n[render_views] Rendering α={alpha:.1f}  ({glb_path})")
            per_alpha_views[alpha] = self.render_four_views(glb_path, alpha_out)

        return stitch_consolidated_views(per_alpha_views, consolidated_dir, self.resolution)


def main():
    parser = argparse.ArgumentParser(
        description="Render 4 views (front/back/left/right) of GLB files, "
                    "optionally consolidating all alpha values into one image per view."
    )
    parser.add_argument(
        "--mode",
        type    = str,
        default = "single",
        choices = ["single", "consolidated"],
        help    = "single: render one or more GLBs independently. "
                  "consolidated: render all alpha_X.X/edit_mesh.glb GLBs and "
                  "stitch into 4 images (one per view direction). (default: single)"
    )

    # ── single-mode args ──────────────────────────────────────────────────
    parser.add_argument(
        "--glb_paths",
        nargs = "+",
        help  = "[single] One or more paths to .glb files"
    )
    parser.add_argument(
        "--front_white",
        type    = str,
        default = None,
        help    = "[single] Path to an already-processed front image. "
                  "When given, the front view is not re-rendered — this image is used directly."
    )

    # ── consolidated-mode args ────────────────────────────────────────────
    parser.add_argument(
        "--alpha_base_dir",
        type = str,
        help = "[consolidated] Directory that contains alpha_X.X/ subdirs, "
               "each with an edit_mesh.glb file."
    )
    parser.add_argument(
        "--consolidated_dir",
        type    = str,
        default = None,
        help    = "[consolidated] Directory to save the 4 consolidated images. "
                  "Defaults to <output_dir>/consolidated."
    )

    # ── shared args ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type     = str,
        required = True,
        help     = "Directory to save rendered images (and per-alpha subdirs in consolidated mode)"
    )
    parser.add_argument(
        "--resolution",
        type    = int,
        default = 512,
        help    = "Render resolution in pixels (default: 512)"
    )
    parser.add_argument(
        "--engine",
        type    = str,
        default = "CYCLES",
        choices = ["CYCLES", "BLENDER_EEVEE"],
        help    = "Blender render engine (default: CYCLES)"
    )

    args = parser.parse_args()
    renderer = GLBViewRenderer(resolution=args.resolution, engine=args.engine)

    # ── consolidated mode ─────────────────────────────────────────────────
    if args.mode == "consolidated":
        if not args.alpha_base_dir:
            parser.error("--alpha_base_dir is required in consolidated mode")
        if not os.path.isdir(args.alpha_base_dir):
            parser.error(f"--alpha_base_dir does not exist: {args.alpha_base_dir}")

        alpha_glb_map = _discover_alpha_glb_map(args.alpha_base_dir)
        print(f"[render_views] Found {len(alpha_glb_map)} alpha GLBs: "
              f"{sorted(alpha_glb_map.keys())}")

        consolidated_dir = args.consolidated_dir or os.path.join(args.output_dir, "consolidated")
        saved = renderer.render_all_alphas_consolidated(
            alpha_glb_map,
            output_dir       = args.output_dir,
            consolidated_dir = consolidated_dir,
        )
        print(f"\n[render_views] Done. Consolidated images ({len(saved)}):")
        for view_name, path in saved.items():
            print(f"  {view_name:5s} → {path}")
        return

    # ── single mode (original behaviour) ─────────────────────────────────
    if not args.glb_paths:
        parser.error("--glb_paths is required in single mode")
    if args.front_white is not None and not os.path.isfile(args.front_white):
        parser.error(f"--front_white path does not exist: {args.front_white}")

    for glb_path in args.glb_paths:
        if not os.path.isfile(glb_path):
            print(f"[render_views] WARNING: file not found, skipping: {glb_path}")
            continue
        print(f"\n[render_views] Processing: {glb_path}")
        saved = renderer.render_four_views(glb_path, args.output_dir, args.front_white)
        print(f"[render_views] Saved {len(saved)} images for {os.path.basename(glb_path)}")


if __name__ == "__main__":
    main()
