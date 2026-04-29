"""
train.py
--------
Trains a lightweight transformer that predicts updated SLAT features
given original (coords, feats) and target deformed coords.

Usage:
    python train.py

Expects PAIRS/ directory from generate_pairs.py
"""

import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import random

# ── Config ────────────────────────────────────────────────────────────────────
PAIRS_DIR   = "PAIRS"
CKPT_DIR    = "CHECKPOINTS"
LOG_EVERY   = 10       # print loss every N steps
SAVE_EVERY  = 5        # save checkpoint every N epochs

FEAT_DIM    = 8        # SLAT feature dimension (confirmed from your run)
COORD_DIM   = 3        # x, y, z
HIDDEN_DIM  = 128
N_HEADS     = 4
N_LAYERS    = 3

BATCH_SIZE  = 1        # one object per batch (variable V size)
LR          = 1e-4
EPOCHS      = 100
MAX_VOXELS  = 8192     # subsample if object has more voxels (memory)
# ──────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────

class SLATPairDataset(Dataset):
    """
    Each item is a (original SLAT, deformed SLAT) pair.
    Returns:
        orig_coords:  [V, 3] float
        orig_feats:   [V, 8] float
        tgt_coords:   [V', 3] float  (deformed — different V possible)
        tgt_feats:    [V', 8] float
        deform_type:  str
    """
    def __init__(self, pairs_dir: str, split: str = 'train', val_ratio: float = 0.15):
        self.pairs = []
        pairs_path = Path(pairs_dir)

        for obj_dir in sorted(pairs_path.iterdir()):
            if not obj_dir.is_dir():
                continue
            orig_path = obj_dir / "slat_orig.pt"
            if not orig_path.exists():
                continue
            for def_path in obj_dir.glob("*.pt"):
                if def_path.name == "slat_orig.pt":
                    continue
                self.pairs.append({
                    'orig':        str(orig_path),
                    'deformed':    str(def_path),
                    'deform_type': def_path.stem,
                })

        # Train/val split
        random.seed(42)
        random.shuffle(self.pairs)
        n_val = max(1, int(len(self.pairs) * val_ratio))
        if split == 'train':
            self.pairs = self.pairs[n_val:]
        else:
            self.pairs = self.pairs[:n_val]

        print(f"Dataset [{split}]: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        orig = torch.load(p['orig'],    map_location='cpu')
        defd = torch.load(p['deformed'], map_location='cpu')

        # Drop batch dim from coords (col 0), keep only x,y,z
        orig_coords = orig['coords'][:, 1:].float()   # [V, 3]
        orig_feats  = orig['feats'].float()            # [V, 8]
        tgt_coords  = defd['coords'][:, 1:].float()   # [V', 3]
        tgt_feats   = defd['feats'].float()            # [V', 8]

        # Normalize coords to [0, 1]
        orig_coords = orig_coords / 63.0
        tgt_coords  = tgt_coords  / 63.0

        # Subsample if too many voxels
        orig_coords, orig_feats = self._subsample(orig_coords, orig_feats)
        tgt_coords,  tgt_feats  = self._subsample(tgt_coords,  tgt_feats)

        return orig_coords, orig_feats, tgt_coords, tgt_feats, p['deform_type']

    def _subsample(self, coords, feats, max_v=MAX_VOXELS):
        V = coords.shape[0]
        if V <= max_v:
            return coords, feats
        idx = torch.randperm(V)[:max_v]
        return coords[idx], feats[idx]


def collate_fn(batch):
    """Keep as list — variable V sizes, no padding needed for now."""
    orig_coords, orig_feats, tgt_coords, tgt_feats, deform_types = zip(*batch)
    return list(orig_coords), list(orig_feats), list(tgt_coords), list(tgt_feats), list(deform_types)


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        # Cross attention
        attn_out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ff(q))
        return q


class DeformationFeatureUpdater(nn.Module):
    """
    Given:
        orig_coords [V, 3]  — original voxel positions
        orig_feats  [V, C]  — original SLAT features
        tgt_coords  [V', 3] — target (deformed) voxel positions

    Predicts:
        updated_feats [V', C] — features consistent with tgt_coords
    """
    def __init__(self, feat_dim=FEAT_DIM, coord_dim=COORD_DIM, hidden=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()

        # Encode source voxels (coord + feat concatenated)
        self.src_encoder = nn.Sequential(
            nn.Linear(coord_dim + feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Encode target coords (geometry query)
        self.tgt_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Stack of cross-attention blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden, n_heads)
            for _ in range(n_layers)
        ])

        # Output projection back to feature space
        self.out_proj = nn.Linear(hidden, feat_dim)

    def forward(self, orig_coords, orig_feats, tgt_coords):
        # orig_coords: [V, 3]
        # orig_feats:  [V, C]
        # tgt_coords:  [V', 3]

        # Source context: encode original voxels
        src = self.src_encoder(
            torch.cat([orig_coords, orig_feats], dim=-1)
        )   # [V, hidden]

        # Query: encode target positions
        tgt = self.tgt_encoder(tgt_coords)   # [V', hidden]

        # Add batch dim for attention
        src = src.unsqueeze(0)   # [1, V,  hidden]
        tgt = tgt.unsqueeze(0)   # [1, V', hidden]

        # Cross-attend: target positions gather from source context
        for layer in self.layers:
            tgt = layer(tgt, src)   # [1, V', hidden]

        tgt = tgt.squeeze(0)         # [V', hidden]
        return self.out_proj(tgt)    # [V', C]


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_ds = SLATPairDataset(PAIRS_DIR, split='train')
    val_ds   = SLATPairDataset(PAIRS_DIR, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model     = DeformationFeatureUpdater().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_losses = []

        for step, (orig_coords_list, orig_feats_list, tgt_coords_list, tgt_feats_list, deform_types) in enumerate(train_loader):
            # Unpack (batch_size=1 so just take index 0)
            orig_coords = orig_coords_list[0].cuda()
            orig_feats  = orig_feats_list[0].cuda()
            tgt_coords  = tgt_coords_list[0].cuda()
            tgt_feats   = tgt_feats_list[0].cuda()

            # Forward
            pred_feats = model(orig_coords, orig_feats, tgt_coords)

            # Loss: L2 between predicted and ground truth deformed features
            loss = F.mse_loss(pred_feats, tgt_feats)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if step % LOG_EVERY == 0:
                print(f"  Epoch {epoch:03d} step {step:04d} | loss={loss.item():.5f} | deform={deform_types[0]}")

        scheduler.step()
        avg_train = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for orig_coords_list, orig_feats_list, tgt_coords_list, tgt_feats_list, _ in val_loader:
                orig_coords = orig_coords_list[0].cuda()
                orig_feats  = orig_feats_list[0].cuda()
                tgt_coords  = tgt_coords_list[0].cuda()
                tgt_feats   = tgt_feats_list[0].cuda()

                pred_feats = model(orig_coords, orig_feats, tgt_coords)
                loss = F.mse_loss(pred_feats, tgt_feats)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.5f} | val_loss={avg_val:.5f} | lr={scheduler.get_last_lr()[0]:.6f}")

        # ── Save checkpoint ──
        if epoch % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': avg_val,
            }, f"{CKPT_DIR}/ckpt_epoch_{epoch:03d}.pt")

        # ── Save best ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': avg_val,
            }, f"{CKPT_DIR}/best.pt")
            print(f"  *** New best val_loss={avg_val:.5f} saved ***")

    print(f"\nTraining done. Best val_loss={best_val_loss:.5f}")
    print(f"Best checkpoint: {CKPT_DIR}/best.pt")


if __name__ == "__main__":
    train()