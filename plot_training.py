"""
plot_training.py — Generate training curves and comparison table from epoch_log.jsonl.

Usage:
    python plot_training.py \
        --log_dir ./output/0/500_0.5 \
        --out_dir ./output/0/500_0.5/plots
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', required=True)
parser.add_argument('--out_dir', required=True)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

log_path = os.path.join(args.log_dir, 'epoch_log.jsonl')
if not os.path.exists(log_path):
    print(f'[ERROR] {log_path} not found')
    exit(1)

records = [json.loads(l) for l in open(log_path)]

epochs      = [r['epoch']      for r in records]
train_loss  = [r['train_loss'] for r in records]

val_epochs     = [r['epoch']      for r in records if r.get('val_loss')   is not None]
val_loss       = [r['val_loss']   for r in records if r.get('val_loss')   is not None]
surf_l2re_vals = [r['surf_l2re'] for r in records if r.get('surf_l2re')  is not None]
vol_l2re_vals  = [r['vol_l2re']  for r in records if r.get('vol_l2re')   is not None]

# ── Fig 1: Training loss ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(epochs, train_loss, label='Train loss (velo + 0.5×press)', color='steelblue')
if val_loss:
    ax.plot(val_epochs, val_loss, label='Val loss', color='orange', marker='o', markersize=3)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Transolver+ — ShapeNet Car: Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(args.out_dir, 'loss_curve.png'), dpi=150)
plt.close(fig)
print(f'Saved loss_curve.png')

# ── Fig 2: Paper metrics (Surf L2RE and Volume L2RE) ───────────────────────
if surf_l2re_vals and vol_l2re_vals:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(val_epochs, surf_l2re_vals, color='crimson', marker='o', markersize=3)
    ax.axhline(0.0745, color='gray', linestyle='--', label='Transolver baseline (0.0745)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative L2')
    ax.set_title('Surf L2RE (surface pressure)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(val_epochs, vol_l2re_vals, color='seagreen', marker='o', markersize=3)
    ax.axhline(0.0207, color='gray', linestyle='--', label='Transolver baseline (0.0207)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative L2')
    ax.set_title('Volume L2RE (surrounding velocity)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Transolver+ — ShapeNet Car: Paper Metrics vs Epoch', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'paper_metrics.png'), dpi=150)
    plt.close(fig)
    print(f'Saved paper_metrics.png')

# ── Fig 3: Log-scale loss (easier to see late-stage behaviour) ─────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(epochs, train_loss, label='Train loss', color='steelblue')
if val_loss:
    ax.semilogy(val_epochs, val_loss, label='Val loss', color='orange', marker='o', markersize=3)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('Transolver+ — ShapeNet Car: Loss (log scale)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(os.path.join(args.out_dir, 'loss_curve_log.png'), dpi=150)
plt.close(fig)
print(f'Saved loss_curve_log.png')

# ── Summary table ───────────────────────────────────────────────────────────
best_surf_epoch = int(np.argmin(surf_l2re_vals)) if surf_l2re_vals else None
best_vol_epoch  = int(np.argmin(vol_l2re_vals))  if vol_l2re_vals  else None

print()
print('=' * 58)
print('TRAINING SUMMARY')
print('=' * 58)
print(f'  Total epochs logged : {len(epochs)}')
print(f'  Final train loss    : {train_loss[-1]:.6f}')
if val_loss:
    print(f'  Best val loss       : {min(val_loss):.6f}  (epoch {val_epochs[int(np.argmin(val_loss))]})')
if surf_l2re_vals:
    best_surf = min(surf_l2re_vals)
    print(f'  Best Surf  L2RE     : {best_surf:.4f}  (epoch {val_epochs[best_surf_epoch]})')
    print(f'  Final Surf L2RE     : {surf_l2re_vals[-1]:.4f}')
if vol_l2re_vals:
    best_vol = min(vol_l2re_vals)
    print(f'  Best Volume L2RE    : {best_vol:.4f}  (epoch {val_epochs[best_vol_epoch]})')
    print(f'  Final Volume L2RE   : {vol_l2re_vals[-1]:.4f}')
print()
print('  Paper Table 3 (Transolver baseline):')
print('    Surf L2RE   = 0.0745')
print('    Volume L2RE = 0.0207')
if surf_l2re_vals and vol_l2re_vals:
    print()
    print('  Your Transolver+ results:')
    print(f'    Surf L2RE   = {min(surf_l2re_vals):.4f}  '
          f'({"better" if min(surf_l2re_vals) < 0.0745 else "worse"} than baseline)')
    print(f'    Volume L2RE = {min(vol_l2re_vals):.4f}  '
          f'({"better" if min(vol_l2re_vals) < 0.0207 else "worse"} than baseline)')
print('=' * 58)

# Write summary JSON
summary = {
    'total_epochs': len(epochs),
    'final_train_loss': train_loss[-1],
    'best_val_loss': min(val_loss) if val_loss else None,
    'best_surf_l2re': min(surf_l2re_vals) if surf_l2re_vals else None,
    'best_vol_l2re':  min(vol_l2re_vals)  if vol_l2re_vals  else None,
    'final_surf_l2re': surf_l2re_vals[-1] if surf_l2re_vals else None,
    'final_vol_l2re':  vol_l2re_vals[-1]  if vol_l2re_vals  else None,
    'paper_baseline': {'surf_l2re': 0.0745, 'vol_l2re': 0.0207},
}
with open(os.path.join(args.out_dir, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)
print(f'\nSaved analysis_summary.json')
