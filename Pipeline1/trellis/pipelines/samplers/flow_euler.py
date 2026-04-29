from typing import *
import torch
import numpy as np
import os
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    # Keys that are sampler-internal and must never be forwarded to the model
    # NOTE: 'mode' and 'step_idx' are intentionally NOT in this set —
    #       they must reach the model's forward() for inversion/edit caching to work.
    _SAMPLER_KEYS = frozenset({'is_head', 'lat_cache'})

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        """
        Base (no-CFG) single-pass model call.
        Forwards mode, step_idx, kv_cache etc. to model.forward() when present.
        Strips only sampler-internal keys (is_head, lat_cache).
        """
        import inspect
        t_tensor = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))

        # Strip sampler-only keys; keep mode, step_idx, kv_cache for the model
        model_kwargs = {k: v for k, v in kwargs.items() if k not in self._SAMPLER_KEYS}

        try:
            sig = inspect.signature(model.forward)
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_var_keyword:
                accepted = set(sig.parameters.keys())
                model_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted}
        except (ValueError, TypeError):
            pass

        return model(x_t, t_tensor, cond, **model_kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    def _base_get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        """
        Always calls FlowEulerSampler._inference_model directly, bypassing any
        CFG / guidance-interval mixin override. Used by invert() so that subclass
        samplers (FlowEulerCfgSampler, FlowEulerGuidanceIntervalSampler) don't
        accidentally require neg_cond / cfg_strength during inversion.
        """
        pred_v = FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(self, model, x_t, t, t_prev, cond=None, **kwargs):
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def invert(self, model, z_0, cond=None, steps=50, rescale_t=1.0, verbose=True, **kwargs):
        """
        DDIM inversion: forward ODE t=0 -> t=1.
        Passes mode='inversion' and step_idx to model.forward() so that
        SparseStructureFlowModel can cache lat_cache / kv_cache per block per step.
        Returns inverted noise z_T.
        """
        sample   = z_0
        load_dir = 'LOAD/'
        os.makedirs(load_dir, exist_ok=True)

        # Forward timestep sequence: 0 -> 1
        t_seq   = np.linspace(0, 1, steps + 1)
        t_seq   = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

        for i, (t, t_next) in enumerate(tqdm(t_pairs, desc="Inverting", disable=not verbose)):
            # Strip CFG-specific keys — inversion is single-pass (no guidance).
            # Keep everything else (including any extra user kwargs).
            clean_kwargs = {k: v for k, v in kwargs.items()
                            if k not in ('neg_cond', 'cfg_strength', 'cfg_interval')}
            # Inject inversion control keys — these MUST reach model.forward()
            clean_kwargs['mode']     = 'inversion'
            clean_kwargs['step_idx'] = i

            # Use _base_get_model_prediction so CFG mixins are bypassed entirely
            pred_x_0, pred_eps, pred_v = self._base_get_model_prediction(
                model, sample, t, cond, **clean_kwargs
            )
            # Forward Euler step: x_{t+dt} = x_t + dt * v
            sample = sample + (t_next - t) * pred_v

        torch.save(sample.detach().cpu().clone(),
                   os.path.join(load_dir, 'inverted_noise.pt'))
        return sample

    @torch.no_grad()
    def sample(self, model, noise, cond=None, steps=50, rescale_t=1.0, verbose=True, **kwargs):
        mode    = kwargs.get('mode', 'normal')
        is_head = kwargs.get('is_head', None)
        device  = noise.device if isinstance(noise, torch.Tensor) else noise.feats.device
        sample  = noise
        load_dir = 'LOAD/'

        # SLAT uses SparseTensor — caching not supported, always normal mode
        if not isinstance(noise, torch.Tensor):
            mode = 'normal'

        t_seq   = np.linspace(1, 0, steps + 1)
        t_seq   = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        ret     = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            step_kwargs = {**kwargs, 'step_idx': i}

            if mode == 'edit':
                # Load per-step latent cache written during inversion
                lat_path = os.path.join(load_dir, f'lat_cache_step_{i:03d}.pt')
                if os.path.exists(lat_path):
                    lat_cache = torch.load(lat_path, map_location=device, weights_only=False)
                    lat_cache = [l.to(device) for l in lat_cache]
                else:
                    lat_cache = None
                    print(f"[warn] lat_cache missing for step {i}")

                # Load per-step KV cache written during inversion
                kv_path = os.path.join(load_dir, f'kv_cache_step_{i:03d}.pt')
                if os.path.exists(kv_path):
                    kv_cache = torch.load(kv_path, map_location=device, weights_only=False)
                    kv_cache = [(k.to(device), v.to(device)) for k, v in kv_cache]
                else:
                    kv_cache = None
                    print(f"[warn] kv_cache missing for step {i}")

                step_kwargs['lat_cache'] = lat_cache
                step_kwargs['kv_cache']  = kv_cache
                step_kwargs['is_head']   = is_head

            out    = self.sample_once(model, sample, t, t_prev, cond, **step_kwargs)

            # Free per-step caches immediately to save VRAM
            if mode == 'edit':
                step_kwargs.pop('lat_cache', None)
                step_kwargs.pop('kv_cache', None)

            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps=50, rescale_t=1.0,
               cfg_strength=3.0, verbose=True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose,
                              neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)

    @torch.no_grad()
    def invert(self, model, z_0, cond, neg_cond=None, steps=50, rescale_t=1.0,
               cfg_strength=3.0, verbose=True, **kwargs):
        # Bypass CFG mixin — inversion is deterministic single-pass
        return FlowEulerSampler.invert(self, model, z_0, cond, steps, rescale_t, verbose, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps=50, rescale_t=1.0,
               cfg_strength=3.0, cfg_interval=(0.0, 1.0), verbose=True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose,
                              neg_cond=neg_cond, cfg_strength=cfg_strength,
                              cfg_interval=cfg_interval, **kwargs)

    @torch.no_grad()
    def invert(self, model, z_0, cond, neg_cond=None, steps=50, rescale_t=1.0,
               verbose=True, **kwargs):
        # Bypass guidance interval mixin — inversion is deterministic single-pass
        return FlowEulerSampler.invert(self, model, z_0, cond, steps, rescale_t, verbose, **kwargs)