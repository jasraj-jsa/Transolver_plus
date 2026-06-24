"""
extract_attn_car.py — Extract slice-to-slice attention maps from a trained
Transolver+ checkpoint on a single ShapeNet Car sample, then save to .npy
and plot a heatmap per layer/head for comparison with Transolver's output.

Usage (no training required):
    python extract_attn_car.py \
        --sample_dir ./dataset/mlcfd_data/car_preprocessed/param0/19f52dd4592c3fb5531e940de4b7770d \
        --checkpoint  ./output/0/200_0.5/checkpoint_best.pth \
        --out_dir     ./results/attn_car \
        --gpu 0
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.Transolver_plus import Model

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', type=str,
                    default='./dataset/mlcfd_data/car_preprocessed/param0/19f52dd4592c3fb5531e940de4b7770d')
parser.add_argument('--checkpoint', type=str,
                    default='./output/0/200_0.5/checkpoint_best.pth')
parser.add_argument('--out_dir',    type=str, default='./results/attn_car')
parser.add_argument('--gpu',        type=str, default='0')
# These must match whatever checkpoint you load
parser.add_argument('--n-layers',   type=int, default=4)
parser.add_argument('--n-hidden',   type=int, default=256)
parser.add_argument('--n-heads',    type=int, default=8)
parser.add_argument('--slice_num',  type=int, default=32)
parser.add_argument('--mlp_ratio',  type=int, default=2)
parser.add_argument('--dropout',    type=float, default=0.1)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
os.makedirs(args.out_dir, exist_ok=True)

# ── Load the single sample directly from .npy files ───────────────────────
sample = args.sample_dir
x    = torch.tensor(np.load(os.path.join(sample, 'x.npy')),   dtype=torch.float)   # (N, 7)
pos  = torch.tensor(np.load(os.path.join(sample, 'pos.npy')), dtype=torch.float)   # (N, 3)
print(f'Sample: {sample}')
print(f'  x shape  : {x.shape}')
print(f'  pos shape: {pos.shape}')

# Add batch dim; condition = zeros(1, 3) as in main_car.py
x_b   = x.unsqueeze(0).to(device)     # (1, N, 7)
pos_b = pos.unsqueeze(0).to(device)   # (1, N, 3)
cond  = torch.zeros(1, 3).to(device)  # (1, 3)

# ── Build model matching the checkpoint ───────────────────────────────────
model = Model(
    space_dim=7,
    n_layers=args.n_layers,
    n_hidden=args.n_hidden,
    dropout=args.dropout,
    n_head=args.n_heads,
    mlp_ratio=args.mlp_ratio,
    fun_dim=0,
    out_dim=4,
    slice_num=args.slice_num,
    unified_pos=0,
).to(device)

# ── Load checkpoint ────────────────────────────────────────────────────────
ckpt_path = args.checkpoint
print(f'Loading checkpoint: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
# Support both plain state_dict and the {model_state_dict: ...} format
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
print('Checkpoint loaded.')

# ── Forward pass (eval, no grad) ──────────────────────────────────────────
model.eval()
with torch.no_grad():
    _ = model((x_b, pos_b, cond))

# ── Collect last_attn from every block ────────────────────────────────────
# Each block.Attn.last_attn has shape (1, H, G, G); squeeze the batch dim.
attn_layers = []
for i, block in enumerate(model.blocks):
    a = block.Attn.last_attn  # (1, H, G, G)
    if a is not None:
        attn_layers.append(a.squeeze(0).cpu().numpy())  # (H, G, G)
    else:
        print(f'  Warning: block {i} has no last_attn (may have been skipped by checkpoint)')

attn_arr = np.stack(attn_layers, axis=0)  # (L, H, G, G)
print(f'Attention array shape: {attn_arr.shape}  (layers, heads, slices, slices)')

# Save the full array
out_npy = os.path.join(args.out_dir, 'car_attn_transolver_plus.npy')
np.save(out_npy, attn_arr)
print(f'Saved: {out_npy}')

n_layers, n_heads, G, _ = attn_arr.shape
ticks = list(range(0, G, 2))  # every-other tick, matches Transolver plot style


def _make_attn_heatmap(attn_2d, title, out_path):
    """Replicate the exact Transolver plot style from the reference screenshot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_2d, cmap='viridis', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel('Keys')
    ax.set_ylabel('Queries')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── 1. Transolver-style: mean over heads, one file per layer ─────────────
for l in range(n_layers):
    mean_attn = attn_arr[l].mean(axis=0)  # (G, G)
    _make_attn_heatmap(
        mean_attn,
        f'Transolver+: Attention among Tokens (Car-ShapeNet) — Layer {l}',
        os.path.join(args.out_dir, f'car_attn_transolver_plus_layer{l}.png')
    )

# ── 2. Side-by-side comparison panel (all layers, mean heads) ────────────
fig, axes = plt.subplots(1, n_layers, figsize=(5.5 * n_layers, 5))
if n_layers == 1:
    axes = [axes]
for l, ax in enumerate(axes):
    mean_attn = attn_arr[l].mean(axis=0)
    im = ax.imshow(mean_attn, cmap='viridis', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel('Keys'); ax.set_ylabel('Queries')
    ax.set_title(f'Layer {l}', fontsize=10)
fig.suptitle(
    f'Transolver+: Attention among Tokens (Car-ShapeNet)\nmean over {n_heads} heads',
    fontsize=12
)
fig.tight_layout()
out_panel = os.path.join(args.out_dir, 'car_attn_transolver_plus_all_layers.png')
fig.savefig(out_panel, dpi=150)
plt.close(fig)
print(f'Saved: {out_panel}')

# ── 3. Per-head grid for each layer ──────────────────────────────────────
cols = min(n_heads, 4)
rows = (n_heads + cols - 1) // cols
for l in range(n_layers):
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for h in range(n_heads):
        im = axes[h].imshow(attn_arr[l, h], cmap='viridis', aspect='auto', vmin=0)
        axes[h].set_title(f'Head {h}', fontsize=9)
        axes[h].set_xticks(ticks); axes[h].set_xticklabels(ticks, fontsize=7)
        axes[h].set_yticks(ticks); axes[h].set_yticklabels(ticks, fontsize=7)
        axes[h].set_xlabel('Keys'); axes[h].set_ylabel('Queries')
        plt.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)
    for h in range(n_heads, len(axes)):
        axes[h].axis('off')
    fig.suptitle(
        f'Transolver+: Attention among Tokens (Car-ShapeNet) — Layer {l}\nper head',
        fontsize=11
    )
    fig.tight_layout()
    out_h = os.path.join(args.out_dir, f'car_attn_transolver_plus_layer{l}_per_head.png')
    fig.savefig(out_h, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_h}')

print(f'\nDone. All outputs in: {args.out_dir}')
