import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import List
from easydict import EasyDict as edict
from PIL import Image

import utils3d
from trellis.utils import render_utils

from groundingdino.util.inference import load_model as gdino_load, predict as gdino_predict
from segment_anything import sam_model_registry, SamPredictor


GDINO_CONFIG  = "/mnt/data/srihari/MODELS/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "/mnt/data/srihari/MODELS/groundingdino_swint_ogc.pth"
SAM_WEIGHTS   = "/mnt/data/srihari/MODELS/sam_vit_h_4b8939.pth"


# ─────────────────────────────────────────────
# 1. Grounded-SAM segmenter
# ─────────────────────────────────────────────

class GroundedSAMSegmenter:
    def __init__(self, device="cuda"):
        self.device = device
        self.gdino  = gdino_load(GDINO_CONFIG, GDINO_WEIGHTS).to(device).eval()
        sam         = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS).to(device)
        self.sam    = SamPredictor(sam)

    @torch.no_grad()
    def segment(
        self,
        image_np:       np.ndarray,
        text_prompt:    str,
        box_threshold:  float = 0.25,
        text_threshold: float = 0.2,
    ) -> np.ndarray:
        import torchvision.transforms.functional as TF

        H, W    = image_np.shape[:2]
        img_pil = Image.fromarray(image_np).convert("RGB")
        img_t   = TF.normalize(
            TF.to_tensor(img_pil),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        boxes, logits, phrases = gdino_predict(
            model          = self.gdino,
            image          = img_t,
            caption        = text_prompt,
            box_threshold  = box_threshold,
            text_threshold = text_threshold,
            device         = self.device,
        )

        if boxes.shape[0] == 0:
            print(f"  GroundingDINO: nothing detected for '{text_prompt}'")
            return np.zeros((H, W), dtype=bool)

        boxes_xyxy       = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

        self.sam.set_image(image_np)
        masks, _, _ = self.sam.predict_torch(
            point_coords     = None,
            point_labels     = None,
            boxes            = self.sam.transform.apply_boxes_torch(boxes_xyxy.to(self.device), (H, W)),
            multimask_output = False,
        )
        mask = masks[:, 0, :, :].any(dim=0).cpu().numpy()
        print(f"  GroundingDINO: {boxes.shape[0]} box(es) for '{text_prompt}', mask coverage {mask.mean()*100:.1f}%")
        return mask


# ─────────────────────────────────────────────
# 2. Cameras — exact match to render_utils.render_video
# ─────────────────────────────────────────────

def get_orbit_cameras(num_views: int, r: float = 2.0, fov: float = 40.0):
    yaws   = torch.linspace(0, 2 * 3.1415, num_views).tolist()
    pitchs = (0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_views))).tolist()
    extrinsics, intrinsics = [], []
    for yaw, pitch in zip(yaws, pitchs):
        fov_t = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw_t = torch.tensor(float(yaw)).cuda()
        pit_t = torch.tensor(float(pitch)).cuda()
        orig  = torch.tensor([
            torch.sin(yaw_t) * torch.cos(pit_t),
            torch.cos(yaw_t) * torch.cos(pit_t),
            torch.sin(pit_t),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 0, 1]).float().cuda(),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t)
        extrinsics.append(extr)
        intrinsics.append(intr)
    return extrinsics, intrinsics


# ─────────────────────────────────────────────
# 3. Unproject 2D mask → 3D voxels
# ─────────────────────────────────────────────

def unproject_mask_to_voxels(
    coords:  torch.Tensor,
    mask_2d: torch.Tensor,
    extr:    torch.Tensor,
    intr:    torch.Tensor,
    img_H:   int,
    img_W:   int,
) -> torch.Tensor:
    N     = coords.shape[0]
    xyz   = (coords[:, 1:].float() / 64.0) - 0.5
    xyz_h = torch.cat([xyz, torch.ones(N, 1, device=xyz.device)], dim=1)

    cam   = (extr @ xyz_h.T).T
    xyz_c = cam[:, :3]

    x  = xyz_c[:, 0] / (xyz_c[:, 2] + 1e-8)
    y  = xyz_c[:, 1] / (xyz_c[:, 2] + 1e-8)
    fx = intr[0, 0];  fy = intr[1, 1]
    cx = intr[0, 2];  cy = intr[1, 2]

    px    = ((x * fx + cx) * img_W).long()
    py    = ((y * fy + cy) * img_H).long()
    valid = (px >= 0) & (px < img_W) & (py >= 0) & (py < img_H) & (xyz_c[:, 2] > 0)

    voxel_mask        = torch.zeros(N, dtype=torch.bool, device=coords.device)
    voxel_mask[valid] = mask_2d[py[valid], px[valid]]
    return voxel_mask


# ─────────────────────────────────────────────
# 4. Multi-view segmentation → voxel mask
# ─────────────────────────────────────────────

def get_voxel_mask_from_text(
    gaussian,
    coords:      torch.Tensor,
    edit_region: str,
    segmenter:   GroundedSAMSegmenter,
    num_views:   int = 8,
    min_votes:   int = 2,
) -> torch.Tensor:
    renders                = render_utils.render_video(gaussian, num_frames=num_views)['color']
    extrinsics, intrinsics = get_orbit_cameras(num_views)
    N                      = coords.shape[0]
    vote_count             = torch.zeros(N, dtype=torch.int32, device=coords.device)

    for i, frame in enumerate(renders):
        H, W    = frame.shape[:2]
        mask_np = segmenter.segment(frame, edit_region)
        mask_t  = torch.from_numpy(mask_np).to(coords.device)

        if mask_t.shape != (H, W):
            mask_t = F.interpolate(
                torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='nearest'
            ).squeeze().bool().to(coords.device)

        vote_count += unproject_mask_to_voxels(
            coords, mask_t, extrinsics[i], intrinsics[i], H, W
        ).int()

    voxel_mask = vote_count >= min_votes
    print(f"Voxel mask: {voxel_mask.sum().item()} / {N} voxels selected for '{edit_region}'")
    return voxel_mask


# ─────────────────────────────────────────────
# 5. Masked sampler injection
# ─────────────────────────────────────────────

@contextmanager
def inject_slat_masked_edit(
    sampler,
    cond_orig: dict,
    cond_edit: dict,
    strength:  float,
    mask:      torch.Tensor,
):
    sampler._old_sample_once = sampler.sample_once

    def _new_sample_once(self_s, model, x_t, t, t_prev, cond, **kwargs):
        out_orig   = sampler._old_sample_once(model, x_t, t, t_prev, cond_orig['cond'], **kwargs)
        out_edit   = sampler._old_sample_once(model, x_t, t, t_prev, cond_edit['cond'], **kwargs)
        s          = (mask.float() * strength).to(x_t.device)[:, None]
        prev_feats = (1 - s) * out_orig.pred_x_prev.feats + s * out_edit.pred_x_prev.feats
        x0_feats   = (1 - s) * out_orig.pred_x_0.feats    + s * out_edit.pred_x_0.feats
        return edict({
            "pred_x_prev": out_orig.pred_x_prev.replace(prev_feats),
            "pred_x_0":    out_orig.pred_x_0.replace(x0_feats),
        })

    sampler.sample_once = _new_sample_once.__get__(sampler, type(sampler))
    yield
    sampler.sample_once = sampler._old_sample_once
    delattr(sampler, '_old_sample_once')


# ─────────────────────────────────────────────
# 6. Top-level pipeline functions
# ─────────────────────────────────────────────

def run_and_save(
    pipeline,
    prompt:                          str,
    num_samples:                     int       = 1,
    seed:                            int       = 42,
    sparse_structure_sampler_params: dict      = {},
    slat_sampler_params:             dict      = {},
    formats:                         List[str] = ['gaussian'],
) -> dict:
    cond   = pipeline.get_cond([prompt])
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
    slat   = pipeline.sample_slat(cond, coords, slat_sampler_params)

    pipeline._saved_coords   = coords
    pipeline._saved_slat     = slat
    pipeline._saved_cond     = cond
    result                   = pipeline.decode_slat(slat, formats)
    pipeline._saved_gaussian = result['gaussian'][0]
    return result


def run_edit_region(
    pipeline,
    segmenter:           GroundedSAMSegmenter,
    edit_region:         str,
    prompt_edit:         str,
    strength:            float     = 0.7,
    seed:                int       = 42,
    num_views:           int       = 8,
    min_votes:           int       = 2,
    slat_sampler_params: dict      = {},
    formats:             List[str] = ['gaussian'],
) -> dict:
    assert hasattr(pipeline, '_saved_coords'), "call run_and_save() first"

    voxel_mask = get_voxel_mask_from_text(
        pipeline._saved_gaussian,
        pipeline._saved_coords,
        edit_region,
        segmenter,
        num_views = num_views,
        min_votes = min_votes,
    )

    cond_edit = pipeline.get_cond([prompt_edit])
    torch.manual_seed(seed)
    with inject_slat_masked_edit(pipeline.slat_sampler, pipeline._saved_cond, cond_edit, strength, voxel_mask):
        slat = pipeline.sample_slat(pipeline._saved_cond, pipeline._saved_coords, slat_sampler_params)

    return pipeline.decode_slat(slat, formats)