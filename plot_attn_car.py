"""
plot_attn_car.py — Replot attention maps from an already-saved .npy file.

Matches the Transolver reference plot style exactly:
  - Single 32x32 heatmap per figure
  - viridis colormap
  - Integer tick labels on Keys / Queries axes
  - Title: "Transolver+: Attention among Tokens (Car-ShapeNet)"

Run locally (no GPU needed):
    python plot_attn_car.py \
        --npy  ./results/attn_car/car_attn_transolver_plus.npy \
        --out_dir ./results/attn_car
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--npy',     type=str,
                    default='./results/attn_car/car_attn_transolver_plus.npy')
parser.add_argument('--out_dir', type=str, default='./results/attn_car')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

attn_arr = np.load(args.npy)          # (L, H, G, G)
print(f'Loaded {args.npy}  shape={attn_arr.shape}')
n_layers, n_heads, G, _ = attn_arr.shape

ticks      = list(range(0, G, 2))
tick_lbls  = [str(t) for t in ticks]


def save_heatmap(attn_2d, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_2d, cmap='viridis', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(ticks);  ax.set_xticklabels(tick_lbls, fontsize=8)
    ax.set_yticks(ticks);  ax.set_yticklabels(tick_lbls, fontsize=8)
    ax.set_xlabel('Keys');  ax.set_ylabel('Queries')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  {path}')


print('\n── Per-layer plots (mean over heads) ──')
for l in range(n_layers):
    mean_attn = attn_arr[l].mean(axis=0)
    save_heatmap(
        mean_attn,
        f'Transolver+: Attention among Tokens (Car-ShapeNet) — Layer {l}',
        os.path.join(args.out_dir, f'attn_layer{l}_mean_heads.png')
    )

print('\n── Per-head plots per layer ──')
cols = min(n_heads, 4)
rows = (n_heads + cols - 1) // cols
for l in range(n_layers):
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for h in range(n_heads):
        im = axes[h].imshow(attn_arr[l, h], cmap='viridis', aspect='auto', vmin=0)
        axes[h].set_title(f'Head {h}', fontsize=9)
        axes[h].set_xticks(ticks);  axes[h].set_xticklabels(tick_lbls, fontsize=7)
        axes[h].set_yticks(ticks);  axes[h].set_yticklabels(tick_lbls, fontsize=7)
        axes[h].set_xlabel('Keys');  axes[h].set_ylabel('Queries')
        plt.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)
    for h in range(n_heads, len(axes)):
        axes[h].axis('off')
    fig.suptitle(
        f'Transolver+: Attention among Tokens (Car-ShapeNet) — Layer {l}',
        fontsize=12
    )
    fig.tight_layout()
    path = os.path.join(args.out_dir, f'attn_layer{l}_per_head.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  {path}')

print('\n── Side-by-side comparison (all layers, mean heads) ──')
fig, axes = plt.subplots(1, n_layers, figsize=(5.5 * n_layers, 5))
if n_layers == 1:
    axes = [axes]
for l, ax in enumerate(axes):
    mean_attn = attn_arr[l].mean(axis=0)
    im = ax.imshow(mean_attn, cmap='viridis', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(ticks);  ax.set_xticklabels(tick_lbls, fontsize=7)
    ax.set_yticks(ticks);  ax.set_yticklabels(tick_lbls, fontsize=7)
    ax.set_xlabel('Keys');  ax.set_ylabel('Queries')
    ax.set_title(f'Layer {l}', fontsize=10)
fig.suptitle(
    f'Transolver+: Attention among Tokens (Car-ShapeNet)  [mean over {n_heads} heads]',
    fontsize=12
)
fig.tight_layout()
path = os.path.join(args.out_dir, 'attn_all_layers_mean_heads.png')
fig.savefig(path, dpi=150)
plt.close(fig)
print(f'  {path}')

print('\n── Paper-style: last layer, all heads stacked vertically (H×G rows × G cols) ──')
last = n_layers - 1
stacked = attn_arr[last].reshape(n_heads * G, G)   # (H*G, G)

fig, ax = plt.subplots(figsize=(5, 8))
im = ax.imshow(stacked, cmap='viridis', aspect='auto', vmin=0)
plt.colorbar(im, ax=ax)

# Draw faint lines separating each head
for h in range(1, n_heads):
    ax.axhline(h * G - 0.5, color='white', linewidth=0.5, alpha=0.4)

# Y ticks: label the start of each head block
ax.set_yticks([h * G for h in range(n_heads)])
ax.set_yticklabels([f'H{h}' for h in range(n_heads)], fontsize=8)
ax.set_xticks(ticks)
ax.set_xticklabels(tick_lbls, fontsize=8)
ax.set_xlabel('Keys')
ax.set_ylabel('Head × Slice (Queries)')
ax.set_title('Transolver+: Attention among Tokens (Car-ShapeNet)\n'
             f'Last layer (layer {last}), all heads stacked', fontsize=11)
fig.tight_layout()
path = os.path.join(args.out_dir, f'attn_layer{last}_stacked_heads.png')
fig.savefig(path, dpi=150)
plt.close(fig)
print(f'  {path}')

print('\nDone.')
