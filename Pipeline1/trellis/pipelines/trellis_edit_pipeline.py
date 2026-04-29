from typing import *
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from .trellis_text_to_3d import TrellisTextTo3DPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Forward (noising) process  — NORMALIZED latent space
#   x_t = (1 - t) * x_0  +  (σ_min + (1 - σ_min) * t) * ε
# ─────────────────────────────────────────────────────────────────────────────

def _q_sample(x_0: torch.Tensor, t: float, sigma_min: float, noise: torch.Tensor) -> torch.Tensor:
    return (1.0 - t) * x_0 + (sigma_min + (1.0 - sigma_min) * t) * noise


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def mask_from_voxel_coords(coords: torch.Tensor, mask_xyz: torch.Tensor) -> torch.Tensor:
    """
    Args:
        coords:   (N, 4) int [batch, x, y, z]
        mask_xyz: (M, 3) int [x, y, z]
    Returns:
        (N,) bool
    """
    mask_set = {tuple(v) for v in mask_xyz.tolist()}
    return torch.tensor(
        [tuple(coords[i, 1:].tolist()) in mask_set for i in range(coords.shape[0])],
        dtype=torch.bool, device=coords.device,
    )


def mask_from_aabb(
    coords: torch.Tensor,
    aabb_min: Tuple[float, float, float],
    aabb_max: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Args:
        coords:   (N, 4) int [batch, x, y, z]
        aabb_min/max: inclusive bounds in voxel-grid space [0, 63]
    Returns:
        (N,) bool
    """
    xyz = coords[:, 1:].float()
    lo  = torch.tensor(aabb_min, device=coords.device, dtype=torch.float32)
    hi  = torch.tensor(aabb_max, device=coords.device, dtype=torch.float32)
    return ((xyz >= lo) & (xyz <= hi)).all(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Core RePaint loop
# ─────────────────────────────────────────────────────────────────────────────

def _repaint_sample_slat(
    flow_model:   torch.nn.Module,
    x_0_known:    torch.Tensor,       # (N, C) NORMALIZED
    coords:       torch.Tensor,       # (N, 4) int
    mask:         torch.Tensor,       # (N,) bool — True = regenerate
    cond:         torch.Tensor,
    neg_cond:     torch.Tensor,
    sigma_min:    float,
    steps:        int,
    rescale_t:    float,
    cfg_strength: float,
    num_resample: int,
    t_start:      float,
    verbose:      bool,
) -> torch.Tensor:
    """
    RePaint denoising for SLatFlow (Stage 2).
    x_0_known must be NORMALIZED (before std*x+mean).
    Returns (N, C) NORMALIZED features — caller must denormalize.
    """
    device = x_0_known.device

    assert not x_0_known.isnan().any(), "x_0_known has NaN before RePaint"
    assert not x_0_known.isinf().any(), "x_0_known has Inf before RePaint"

    t_seq   = np.linspace(1, 0, steps + 1)
    t_seq   = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
    t_seq   = t_seq[t_seq <= t_start]
    if len(t_seq) == 0 or float(t_seq[0]) < t_start:
        t_seq = np.concatenate([[t_start], t_seq])
    t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

    x_t = _q_sample(x_0_known, t_start, sigma_min, torch.randn_like(x_0_known))

    def _sp(feats): return sp.SparseTensor(feats=feats, coords=coords)

    def _cfg_v(x_t, t):
        t_ten = torch.tensor([1000.0 * t], device=device, dtype=torch.float32)
        v_c   = flow_model(_sp(x_t), t_ten, cond)
        v_u   = flow_model(_sp(x_t), t_ten, neg_cond)
        if isinstance(v_c, sp.SparseTensor): v_c = v_c.feats
        if isinstance(v_u, sp.SparseTensor): v_u = v_u.feats
        return v_u + cfg_strength * (v_c - v_u)

    for t, t_prev in tqdm(t_pairs, desc="RePaint SLatFlow", disable=not verbose):
        for r in range(num_resample):
            pred        = x_t - (t - t_prev) * _cfg_v(x_t, t)
            x_kn_tp     = _q_sample(x_0_known, t_prev, sigma_min, torch.randn_like(x_0_known))
            pred[~mask] = x_kn_tp[~mask]

            if r < num_resample - 1:
                x_t_new          = _q_sample(pred,      t, sigma_min, torch.randn_like(x_0_known))
                x_kn_t           = _q_sample(x_0_known, t, sigma_min, torch.randn_like(x_0_known))
                x_t_new[~mask]   = x_kn_t[~mask]
                x_t              = x_t_new
            else:
                x_t = pred

    return x_t


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TrellisTextTo3DEditingPipeline(TrellisTextTo3DPipeline):
    """
    Extends TrellisTextTo3DPipeline with RePaint-based region editing.

    Normalization contract
    ─────────────────────
    sample_slat() ends with:   feats = feats * std + mean   (denormalize)
    So slat.feats is DENORMALIZED when it comes out of sample_slat().
    This pipeline inverts that before the flow model and re-applies after.
    std/mean are per-channel vectors of shape (C,) — we reshape to (1, C)
    for correct broadcasting against (N, C) feature tensors.
    """

    mask_from_voxel_coords = staticmethod(mask_from_voxel_coords)
    mask_from_aabb         = staticmethod(mask_from_aabb)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DEditingPipeline":
        from .base import Pipeline as _Base
        base = _Base.from_pretrained(path)
        args = base._pretrained_args

        p = TrellisTextTo3DEditingPipeline()
        p.__dict__ = base.__dict__

        p.sparse_structure_sampler = getattr(
            samplers, args['sparse_structure_sampler']['name']
        )(**args['sparse_structure_sampler']['args'])
        p.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        p.slat_sampler = getattr(
            samplers, args['slat_sampler']['name']
        )(**args['slat_sampler']['args'])
        p.slat_sampler_params = args['slat_sampler']['params']

        p.slat_normalization = args['slat_normalization']
        p._init_text_cond_model(args['text_cond_model'])

        return p

    # ── Normalization ─────────────────────────────────────────────────────────

    def _normalize(self, feats: torch.Tensor) -> torch.Tensor:
        """Denormalized (N,C) → normalized (N,C):  x = (feats - mean) / std"""
        std  = torch.tensor(self.slat_normalization['std'],
                            device=feats.device, dtype=feats.dtype).reshape(1, -1)
        mean = torch.tensor(self.slat_normalization['mean'],
                            device=feats.device, dtype=feats.dtype).reshape(1, -1)
        return (feats - mean) / (std + 1e-8)

    def _denormalize(self, feats: torch.Tensor) -> torch.Tensor:
        """Normalized (N,C) → denormalized (N,C):  feats = x * std + mean"""
        std  = torch.tensor(self.slat_normalization['std'],
                            device=feats.device, dtype=feats.dtype).reshape(1, -1)
        mean = torch.tensor(self.slat_normalization['mean'],
                            device=feats.device, dtype=feats.dtype).reshape(1, -1)
        return feats * std + mean

    @property
    def _sigma_min(self) -> float:
        return self.slat_sampler.sigma_min

    @property
    def _sparse_sigma_min(self) -> float:
        return self.sparse_structure_sampler.sigma_min

    # ── Appearance-only edit ──────────────────────────────────────────────────

    @torch.no_grad()
    def run_edit_appearance(
        self,
        slat_known:   sp.SparseTensor,
        mask:         torch.Tensor,
        prompt:       str,
        steps:        int   = 50,
        cfg_strength: float = 7.5,
        num_resample: int   = 1,
        t_start:      float = 1.0,
        rescale_t:    float = 1.0,
        formats:      List[str] = ["mesh", "gaussian", "radiance_field"],
        verbose:      bool  = True,
    ) -> dict:
        """
        Retexture masked voxels without changing geometry.
        slat_known must be the denormalized output of sample_slat().
        mask must be aligned to slat_known.coords.
        """
        cond     = self.get_cond([prompt])
        neg_cond = self.text_cond_model["null_cond"]

        feats = slat_known.feats
        print(f"  [run_edit_appearance] input feats: shape={feats.shape} "
              f"min={feats.min():.4f} max={feats.max():.4f} "
              f"nan={feats.isnan().sum().item()} inf={feats.isinf().sum().item()}")

        # Normalize into flow-model space
        x_0_norm = self._normalize(feats)
        print(f"  [run_edit_appearance] normalized: "
              f"min={x_0_norm.min():.4f} max={x_0_norm.max():.4f}")

        edited_norm = _repaint_sample_slat(
            flow_model   = self.models["slat_flow_model"],
            x_0_known    = x_0_norm,
            coords       = slat_known.coords,
            mask         = mask,
            cond         = cond["cond"],
            neg_cond     = neg_cond,
            sigma_min    = self._sigma_min,
            steps        = steps,
            rescale_t    = rescale_t,
            cfg_strength = cfg_strength,
            num_resample = num_resample,
            t_start      = t_start,
            verbose      = verbose,
        )

        # Denormalize back into decoder space
        edited_feats = self._denormalize(edited_norm)
        print(f"  [run_edit_appearance] decoded feats: "
              f"min={edited_feats.min():.4f} max={edited_feats.max():.4f} "
              f"nan={edited_feats.isnan().sum().item()} inf={edited_feats.isinf().sum().item()}")

        # Hard clamp — catches any extreme outliers from the flow model that
        # would produce astronomically large Gaussian scales and OOM the renderer.
        # The valid range mirrors what sample_slat() produces after denorm.
        ref_min = feats.min().item()
        ref_max = feats.max().item()
        edited_feats = edited_feats.clamp(ref_min * 3.0, ref_max * 3.0)

        edited_slat = sp.SparseTensor(feats=edited_feats, coords=slat_known.coords)
        return self.decode_slat(edited_slat, formats)

    # ── Geometry + appearance edit ────────────────────────────────────────────

    @torch.no_grad()
    def run_edit_geometry(
        self,
        mesh:         o3d.geometry.TriangleMesh,
        slat_known:   sp.SparseTensor,
        mask:         torch.Tensor,
        prompt:       str,
        steps:        int   = 50,
        cfg_strength: float = 7.5,
        num_resample: int   = 1,
        t_start:      float = 1.0,
        rescale_t:    float = 1.0,
        formats:      List[str] = ["mesh", "gaussian", "radiance_field"],
        verbose:      bool  = True,
    ) -> dict:
        cond     = self.get_cond([prompt])
        neg_cond = self.text_cond_model["null_cond"]

        new_coords     = self._repaint_sparse_structure(
            mesh=mesh, mask=mask, cond=cond,
            steps=steps, rescale_t=rescale_t, cfg_strength=cfg_strength,
            num_resample=num_resample, t_start=t_start, verbose=verbose,
        )
        new_slat_known = self._remap_slat_to_new_coords(slat_known, mask, new_coords)
        new_mask       = self._build_new_mask(new_coords, slat_known.coords, mask)

        x_0_norm    = self._normalize(new_slat_known.feats)
        edited_norm = _repaint_sample_slat(
            flow_model   = self.models["slat_flow_model"],
            x_0_known    = x_0_norm,
            coords       = new_coords,
            mask         = new_mask,
            cond         = cond["cond"],
            neg_cond     = neg_cond,
            sigma_min    = self._sigma_min,
            steps        = steps,
            rescale_t    = rescale_t,
            cfg_strength = cfg_strength,
            num_resample = num_resample,
            t_start      = t_start,
            verbose      = verbose,
        )

        edited_feats = self._denormalize(edited_norm)
        ref_min = slat_known.feats.min().item()
        ref_max = slat_known.feats.max().item()
        edited_feats = edited_feats.clamp(ref_min * 3.0, ref_max * 3.0)

        return self.decode_slat(
            sp.SparseTensor(feats=edited_feats, coords=new_coords), formats
        )

    # ── SSFlow RePaint (Stage 1) ──────────────────────────────────────────────

    def _repaint_sparse_structure(
        self,
        mesh, mask, cond,
        steps, rescale_t, cfg_strength, num_resample, t_start, verbose,
    ):
        flow_model    = self.models["sparse_structure_flow_model"]
        decoder       = self.models["sparse_structure_decoder"]
        reso          = flow_model.resolution
        sigma_min     = self._sparse_sigma_min
        device        = self.device
        coords_sparse = self.voxelize(mesh)
        C             = flow_model.in_channels

        x_0 = torch.zeros(1, C, reso, reso, reso, device=device)
        x_0[0, :, coords_sparse[:, 0], coords_sparse[:, 1], coords_sparse[:, 2]] = 1.0

        mask_dense = torch.zeros(1, C, reso, reso, reso, dtype=torch.bool, device=device)
        masked_xyz = coords_sparse[mask]
        if masked_xyz.shape[0] > 0:
            mask_dense[0, :, masked_xyz[:, 0], masked_xyz[:, 1], masked_xyz[:, 2]] = True

        t_seq   = np.linspace(1, 0, steps + 1)
        t_seq   = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_seq   = t_seq[t_seq <= t_start]
        if len(t_seq) == 0 or float(t_seq[0]) < t_start:
            t_seq = np.concatenate([[t_start], t_seq])
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

        x_t = _q_sample(x_0, t_start, sigma_min, torch.randn_like(x_0))

        for t, t_prev in tqdm(t_pairs, desc="RePaint SSFlow", disable=not verbose):
            for r in range(num_resample):
                t_ten             = torch.tensor([1000.0 * t], device=device, dtype=torch.float32)
                v_c               = flow_model(x_t, t_ten, cond["cond"])
                v_u               = flow_model(x_t, t_ten, cond["neg_cond"])
                pred              = x_t - (t - t_prev) * (v_u + cfg_strength * (v_c - v_u))
                x_kn_tp           = _q_sample(x_0, t_prev, sigma_min, torch.randn_like(x_0))
                pred[~mask_dense] = x_kn_tp[~mask_dense]

                if r < num_resample - 1:
                    x_t_new              = _q_sample(pred, t, sigma_min, torch.randn_like(x_0))
                    x_kn_t               = _q_sample(x_0,  t, sigma_min, torch.randn_like(x_0))
                    x_t_new[~mask_dense] = x_kn_t[~mask_dense]
                    x_t                  = x_t_new
                else:
                    x_t = pred

        return torch.argwhere(decoder(x_t) > 0)[:, [0, 2, 3, 4]].int()

    # ── Coord remapping ───────────────────────────────────────────────────────

    def _remap_slat_to_new_coords(self, slat_known, old_mask, new_coords):
        old_coords = slat_known.coords
        old_feats  = slat_known.feats
        C, device  = old_feats.shape[1], old_feats.device
        lookup = {
            tuple(old_coords[i, 1:].tolist()): old_feats[i]
            for i in range(old_coords.shape[0]) if not old_mask[i]
        }
        new_feats = torch.zeros(new_coords.shape[0], C, device=device, dtype=old_feats.dtype)
        for i, c in enumerate(new_coords.tolist()):
            k = tuple(c[1:])
            if k in lookup:
                new_feats[i] = lookup[k]
        return sp.SparseTensor(feats=new_feats, coords=new_coords)

    def _build_new_mask(self, new_coords, old_coords, old_mask):
        old_unmasked = {
            tuple(old_coords[i, 1:].tolist())
            for i in range(old_coords.shape[0]) if not old_mask[i]
        }
        return torch.tensor(
            [tuple(c[1:]) not in old_unmasked for c in new_coords.tolist()],
            dtype=torch.bool, device=new_coords.device,
        )

    @torch.no_grad()
    def encode_mesh_to_slat(self, mesh):
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=self.device),
            coords,
        ], dim=1)
        return coords, self.sample_slat(self.get_cond([""]), coords, {})