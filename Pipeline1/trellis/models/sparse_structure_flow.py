from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=torch.device('cuda')) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self):
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _get_kv(self, block, h, t_emb, cond):
        """
        Extract the K and V tensors that self-attention would compute for tokens h.
        Returns k, v each of shape [B, N, H, C//H].
        """
        if self.share_mod:
            shift_msa, scale_msa, _, _, _, _ = t_emb.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, _, _, _, _ = block.adaLN_modulation(t_emb).chunk(6, dim=1)
        h_norm = block.norm1(h)
        h_norm = h_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv = block.self_attn.to_qkv(h_norm)           # [B, N, 3*C]
        B, N, _ = qkv.shape
        C = self.model_channels
        H = self.num_heads
        qkv = qkv.reshape(B, N, 3, H, C // H)          # [B, N, 3, H, C//H]
        _, k, v = qkv.unbind(dim=2)                     # each [B, N, H, C//H]
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        is_head: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        lat_cache: Optional[List[torch.Tensor]] = None,
        mode: str = 'normal',
        step_idx: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        mode='normal'    : standard forward, no caching
        mode='inversion' : forward pass — cache K/V and latent h per block,
                           save to LOAD/kv_cache_step_NNN.pt and
                                     LOAD/lat_cache_step_NNN.pt
        mode='edit'      : VoxHammer editing pass —
                             head tokens evolve freely under new (steered) cond,
                             body tokens use cached K/V for attention context and
                             cached latent (hard replacement) after each block,
                             freezing body structure to the inversion trajectory.
        """
        # ── Tokenization ────────────────────────────────────────────────────
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        # ── Embedding ───────────────────────────────────────────────────────
        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        t_emb = t_emb.type(self.dtype)
        h     = h.type(self.dtype)
        cond  = cond.type(self.dtype)

        step_kv_cache  = []   # collected during inversion, one entry per block
        step_lat_cache = []   # collected during inversion, one entry per block

        for i, block in enumerate(self.blocks):

            if mode == 'edit' and kv_cache is not None and lat_cache is not None:
                # ────────────────────────────────────────────────────────────
                # VoxHammer edit pass:
                #
                # 1. Inject cached body K/V into self-attention so that head
                #    tokens attend to the faithful body context from inversion.
                # 2. Run the block normally (head tokens evolve under new cond).
                # 3. Hard-replace body token activations with the cached
                #    inversion latent — body structure is frozen.
                # ────────────────────────────────────────────────────────────
                cached_k, cached_v = kv_cache[i]        # [1, N, H, C//H]

                # Expand for CFG batch (cond + uncond = 2)
                B = h.shape[0]
                if cached_k.shape[0] < B:
                    cached_k = cached_k.expand(B, -1, -1, -1).contiguous()
                    cached_v = cached_v.expand(B, -1, -1, -1).contiguous()

                # Hook: swap body K/V with cached values before SDPA
                hook_handle = self._register_kv_inject_hook(
                    block.self_attn, cached_k, cached_v, is_head
                )
                h = block(h, t_emb, cond)
                hook_handle.remove()

                # Hard-replace body token activations with cached inversion latent
                cached_lat = lat_cache[i].to(h.device).to(h.dtype)  # [1, N, C]
                if cached_lat.shape[0] < B:
                    cached_lat = cached_lat.expand(B, -1, -1).contiguous()
                body = ~is_head   # boolean mask [N]
                h[:, body, :] = cached_lat[:, body, :]

            elif mode == 'inversion':
                h = block(h, t_emb, cond)
                # Cache K/V (for injection during edit) and full latent
                k, v = self._get_kv(block, h, t_emb, cond)
                step_kv_cache.append((k.detach().cpu(), v.detach().cpu()))
                step_lat_cache.append(h.detach().cpu())

            else:  # normal
                h = block(h, t_emb, cond)

        # ── Save inversion cache to disk (one file per timestep step) ───────
        if mode == 'inversion' and step_idx is not None:
            save_dir = 'LOAD/'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(step_kv_cache,
                       os.path.join(save_dir, f'kv_cache_step_{step_idx:03d}.pt'))
            torch.save(step_lat_cache,
                       os.path.join(save_dir, f'lat_cache_step_{step_idx:03d}.pt'))

        # ── Output projection ────────────────────────────────────────────────
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(
            h.shape[0], h.shape[2],
            *[self.resolution // self.patch_size] * 3
        )
        h = unpatchify(h, self.patch_size).contiguous()
        return h

    def _register_kv_inject_hook(self, self_attn_module, cached_k, cached_v, is_head):
        """
        Register a forward pre-hook on the self-attention module that replaces
        the K and V projections of body tokens (~is_head) with the cached values
        before SDPA runs.

        cached_k, cached_v : [B, N, H, C//H]
        is_head            : bool [N] — True for tokens in the editable head region
        """
        C  = self.model_channels
        H  = self.num_heads
        Ch = C // H

        class _HookOutput(Exception):
            def __init__(self, val):
                self.val = val

        def hook(module, args, kwargs_hook):
            x_in = args[0]          # [B, N, C]
            B, N, _ = x_in.shape

            # Fresh QKV from current (steered) activations
            qkv = module.to_qkv(x_in)              # [B, N, 3*C]
            qkv = qkv.reshape(B, N, 3, H, Ch)
            q, k, v = qkv.unbind(dim=2)             # each [B, N, H, Ch]

            # Overwrite body positions with cached inversion K/V
            body = ~is_head                          # [N]
            k[:, body, :, :] = cached_k[:, body, :, :].to(k.dtype)
            v[:, body, :, :] = cached_v[:, body, :, :].to(v.dtype)

            # Run SDPA manually (permute to [B, H, N, Ch])
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v)   # [B, H, N, Ch]
            out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # [B, N, C]
            out = module.to_out(out)
            raise _HookOutput(out)

        # Wrap forward so the hook's output is used
        orig_forward = self_attn_module.forward

        def patched_forward(*a, **kw):
            try:
                return orig_forward(*a, **kw)
            except _HookOutput as e:
                return e.val

        self_attn_module.forward = patched_forward
        handle = self_attn_module.register_forward_pre_hook(hook, with_kwargs=True)

        class _CleanupHandle:
            def remove(self_inner):
                handle.remove()
                self_attn_module.forward = orig_forward

        return _CleanupHandle()