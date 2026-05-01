#!/usr/bin/env python3
"""
render_views.py — Render front, back, left, and right views of one or more GLB files.

Usage:
    python3 render_views.py --glb_paths model1.glb model2.glb --output_dir ./views
    python3 render_views.py --glb_paths ./output/src_mesh.glb --output_dir ./views --resolution 512
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


class GLBViewRenderer:
    def __init__(self, resolution: int = 512, engine: str = "CYCLES"):
        self.resolution = resolution
        self.engine = engine

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _clear_scene(self):
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for mat in bpy.data.materials:
            bpy.data.materials.remove(mat, do_unlink=True)
        for tex in bpy.data.textures:
            bpy.data.textures.remove(tex, do_unlink=True)
        for img in bpy.data.images:
            bpy.data.images.remove(img, do_unlink=True)

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


def main():
    parser = argparse.ArgumentParser(description="Render 4 views (front/back/left/right) of GLB files")
    parser.add_argument(
        "--glb_paths",
        nargs    = "+",
        required = True,
        help     = "One or more paths to .glb files"
    )
    parser.add_argument(
        "--output_dir",
        type     = str,
        required = True,
        help     = "Directory to save rendered images"
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
    parser.add_argument(
        "--front_white",
        type    = str,
        default = None,
        help    = "Path to an already-processed front image (e.g. output/image/front_white.png). "
                  "When given, the front view is not re-rendered — this image is used directly."
    )
    args = parser.parse_args()

    if args.front_white is not None and not os.path.isfile(args.front_white):
        parser.error(f"--front_white path does not exist: {args.front_white}")

    renderer = GLBViewRenderer(resolution=args.resolution, engine=args.engine)

    for glb_path in args.glb_paths:
        if not os.path.isfile(glb_path):
            print(f"[render_views] WARNING: file not found, skipping: {glb_path}")
            continue
        print(f"\n[render_views] Processing: {glb_path}")
        saved = renderer.render_four_views(glb_path, args.output_dir, args.front_white)
        print(f"[render_views] Saved {len(saved)} images for {os.path.basename(glb_path)}")


if __name__ == "__main__":
    main()
