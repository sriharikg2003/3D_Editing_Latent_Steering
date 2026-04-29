import os
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

# ─── Config ───────────────────────────────────────────────────────────────────
PIPELINE_PATH  = "/mnt/data/srihari/MODELS/TRELLIS-image-large"
IMAGE_PATH     = "/mnt/data/srihari/my_TRELLIS/girl.avif"
NUM_DIRECTIONS = 12
ITERATIONS     = 500
MIN_EPS        = 0.5
MAX_EPS        = 3.0
LAMBDA_REG     = 0.25
SAVE_OBJ_EVERY = 50
VOXEL_RESO     = 16
CKPT_DIR       = "checkpoints"
RESULTS_DIR    = "discovery_results"
PLOT_PATH      = "discovery_training_curves.png"
# ──────────────────────────────────────────────────────────────────────────────


class StructureReconstructor(nn.Module):
    def __init__(self, num_directions, feature_dim=VOXEL_RESO**3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )
        self.cls_head = nn.Linear(512, num_directions)
        self.reg_head = nn.Linear(512, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.cls_head(h), self.reg_head(h).squeeze(-1)


def get_structural_features(coords, reso=VOXEL_RESO):
    voxels = torch.zeros((1, 1, reso, reso, reso), device=coords.device)
    if coords.shape[0] > 0:
        idx = torch.clamp(coords[:, 1:].long(), 0, reso - 1)
        voxels[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return voxels.view(1, -1)


def save_plot(history, path=PLOT_PATH):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["loss"],     color='tab:blue',   lw=1.2)
    axes[0].set_title("Total Loss");   axes[0].set_xlabel("Iteration")
    axes[1].plot(history["cls_loss"], color='tab:red',    lw=1.2)
    axes[1].set_title("Cls Loss");     axes[1].set_xlabel("Iteration")
    axes[2].plot(history["accuracy"], color='tab:orange', lw=1.2)
    axes[2].set_title("Accuracy");     axes[2].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def extract_tensor_from_result(result):
    """
    The sampler returns an EasyDict. Find the tensor that is the denoised
    latent — it will be the largest float tensor in the dict.
    Also print keys once so we know the structure.
    """
    if isinstance(result, torch.Tensor):
        return result

    if hasattr(result, 'keys'):           # EasyDict / dict
        print(f"  [hook] sampler returned EasyDict with keys: {list(result.keys())}")
        # try common key names first
        for key in ['samples', 'sample', 'x', 'z', 'latent', 'pred', 'x_0', 'x0']:
            if key in result and isinstance(result[key], torch.Tensor):
                print(f"  [hook] using key='{key}', shape={result[key].shape}")
                return result[key]
        # fallback: largest float tensor
        best = None
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                if best is None or v.numel() > best.numel():
                    best = v
        if best is not None:
            print(f"  [hook] fallback: using largest tensor, shape={best.shape}")
            return best

    raise ValueError(f"Cannot extract tensor from sampler result: {type(result)}, keys={list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")


def capture_z_base(pipeline, image):
    """
    Run pipeline.run() once, hook sampler.sample() to grab the final
    denoised latent z_base without touching the sampler's arguments.
    """
    captured = {}
    _orig = pipeline.sparse_structure_sampler.sample

    def _hook(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if 'z' not in captured:          # only capture once (first call)
            z = extract_tensor_from_result(result)
            captured['z'] = z.detach().clone()
        return result

    pipeline.sparse_structure_sampler.sample = _hook
    with torch.no_grad():
        outputs = pipeline.run(image, seed=1)
    pipeline.sparse_structure_sampler.sample = _orig

    assert 'z' in captured, "Hook never fired — sampler.sample() was not called"
    return outputs, captured['z']


def decode_latent_to_coords(pipeline, z):
    """
    Single-step decode: sparse structure decoder (conv3d) → occupancy → coords.
    No diffusion, no CFG — direct structural decode.
    """
    with torch.no_grad():
        # Try both common key names for the decoder
        dec = pipeline.models.get('sparse_structure_decoder',
              pipeline.models.get('ss_dec', None))
        if dec is None:
            # print available keys once for debugging
            print(f"  [decode] available model keys: {list(pipeline.models.keys())}")
            raise KeyError("Cannot find sparse structure decoder in pipeline.models")

        logits = dec(z)          # (1, 1, R, R, R) or (1, C, R, R, R)
        occ    = logits[:, 0]    # take first channel as occupancy
        coords = torch.argwhere(occ[0] > 0).float()   # (N, 3)
        if coords.shape[0] < 8:
            flat  = occ[0].reshape(-1)
            R     = occ.shape[-1]
            topk  = torch.topk(flat, min(512, flat.numel())).indices
            coords = torch.stack([topk // R**2, (topk // R) % R, topk % R], 1).float()
        batch = torch.zeros(coords.shape[0], 1, device=z.device)
        return torch.cat([batch, coords], dim=1)


def render_and_save(pipeline, cond, coords, base_slat_rng_state, out_path):
    torch.cuda.set_rng_state(base_slat_rng_state)
    slat    = pipeline.sample_slat(cond, coords)
    outputs = pipeline.decode_slat(slat, formats=['gaussian'])
    video   = render_utils.render_video(outputs['gaussian'][0])['color']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, video, fps=30)


def save_all_directions(pipeline, cond, z_base, base_slat_rng_state, A, iter_tag):
    out_dir = os.path.join(RESULTS_DIR, iter_tag)
    os.makedirs(out_dir, exist_ok=True)
    c, r = z_base.shape[1], z_base.shape[2]
    for d_idx in range(NUM_DIRECTIONS):
        dir_vec = A[d_idx].view(1, c, r, r, r)
        z_edit  = z_base + MAX_EPS * dir_vec
        coords  = decode_latent_to_coords(pipeline, z_edit)
        render_and_save(pipeline, cond, coords, base_slat_rng_state,
                        os.path.join(out_dir, f"dir_{d_idx:02d}.mp4"))
    print(f"  → {NUM_DIRECTIONS} renders saved to {out_dir}/")


def main():
    os.makedirs(CKPT_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(PIPELINE_PATH)
    pipeline.cuda()

    image = Image.open(IMAGE_PATH)

    # ── Run once, capture z_base ───────────────────────────────────────────
    print("Running base pipeline + capturing z_base...")
    outputs_base, z_base = capture_z_base(pipeline, image)
    print(f"  z_base shape: {z_base.shape}")   # expect (1, 8, 16, 16, 16)

    c, r       = z_base.shape[1], z_base.shape[2]
    latent_dim = c * r * r * r

    base_dir = os.path.join(RESULTS_DIR, "base")
    os.makedirs(base_dir, exist_ok=True)
    base_video = render_utils.render_video(outputs_base['gaussian'][0])['color']
    imageio.mimsave(os.path.join(base_dir, "render.mp4"), base_video, fps=30)
    print(f"  → base render saved to {base_dir}/render.mp4")

    # ── Image cond + freeze appearance RNG ────────────────────────────────
    with torch.no_grad():
        cond = pipeline.get_cond([image])

    base_coords = decode_latent_to_coords(pipeline, z_base)
    f_base      = get_structural_features(base_coords)

    torch.manual_seed(42)
    base_slat_rng_state = torch.cuda.get_rng_state()

    # ── Trainable directions ───────────────────────────────────────────────
    A = nn.Parameter(torch.zeros(NUM_DIRECTIONS, latent_dim).cuda())
    nn.init.orthogonal_(A.data)
    A.data *= 0.1

    reconstructor = StructureReconstructor(NUM_DIRECTIONS).cuda()
    optimizer     = optim.Adam([
        {'params': [A],                        'lr': 5e-3},
        {'params': reconstructor.parameters(), 'lr': 1e-3},
    ])
    cls_criterion = nn.CrossEntropyLoss()

    print("\nSaving iter_0000 (untrained directions)...")
    with torch.no_grad():
        save_all_directions(pipeline, cond, z_base, base_slat_rng_state,
                             A, iter_tag="iter_0000")

    history = {"loss": [], "cls_loss": [], "reg_loss": [], "accuracy": []}

    print(f"\nTraining {ITERATIONS} iters | {NUM_DIRECTIONS} dirs "
          f"| eps∈[{MIN_EPS},{MAX_EPS}] | latent_dim={latent_dim}")

    for i in tqdm(range(ITERATIONS)):
        optimizer.zero_grad()

        k   = torch.randint(0, NUM_DIRECTIONS, (1,)).cuda()
        eps = (torch.rand(1) * (MAX_EPS - MIN_EPS) + MIN_EPS).cuda()

        dir_vec = A[k].view(1, c, r, r, r)
        z_edit  = z_base + eps * dir_vec

        with torch.no_grad():
            coords_edit = decode_latent_to_coords(pipeline, z_edit)

        f_edit = get_structural_features(coords_edit)
        diff   = f_edit - f_base

        logits_k, pred_eps = reconstructor(diff)
        loss_cls = cls_criterion(logits_k, k)
        loss_reg = torch.abs(pred_eps - eps).mean()
        loss     = loss_cls + LAMBDA_REG * loss_reg
        acc      = (torch.argmax(logits_k, dim=1) == k).float().mean()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            A.data = A.data / (A.data.norm(dim=1, keepdim=True) + 1e-8)

        history["loss"].append(loss.item())
        history["cls_loss"].append(loss_cls.item())
        history["reg_loss"].append(loss_reg.item())
        history["accuracy"].append(acc.item())
        save_plot(history)

        if (i + 1) % SAVE_OBJ_EVERY == 0:
            iter_tag = f"iter_{i+1:04d}"
            print(f"\n[{i+1}] Checkpoint → {iter_tag}/")
            with torch.no_grad():
                save_all_directions(pipeline, cond, z_base, base_slat_rng_state,
                                     A, iter_tag=iter_tag)
            torch.save(A.detach().cpu(),
                       os.path.join(CKPT_DIR, f"directions_A_{iter_tag}.pt"))

    torch.save(A.detach().cpu(), os.path.join(CKPT_DIR, "directions_A_final.pt"))
    torch.save(reconstructor.state_dict(), os.path.join(CKPT_DIR, "reconstructor_final.pt"))
    save_plot(history)

    print("\nGenerating final direction renders...")
    with torch.no_grad():
        save_all_directions(pipeline, cond, z_base, base_slat_rng_state,
                             A, iter_tag="final")

    print(f"\nDone.\n  Results → {RESULTS_DIR}/\n  Ckpts   → {CKPT_DIR}/")


if __name__ == "__main__":
    main()