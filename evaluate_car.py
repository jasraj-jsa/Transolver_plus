"""
Evaluation script for Transolver+ on ShapeNet Car (preprocessed .npy data).

Metrics (all computed in NORMALIZED space, matching paper convention):
  - Surf   : relative L2 on surface pressure nodes    (last channel, surf_mask==True)
  - Volume : relative L2 on surrounding velocity nodes (first 3 channels, surf_mask==False)

The preprocessed .npy files already contain NORMALIZED y values (get_datalist applied
coef_norm in-place before saving). The model also outputs in normalized space. So no
de-normalization is needed — both sides are in the same space.

CD / rho_D are skipped because they require raw .vtk files not present on Kaggle.

Usage:
    python evaluate_car.py \\
        --save_dir /kaggle/input/datasets/nikshiith/shapenetcar-preprocessed/car_preprocessed \\
        --checkpoint /kaggle/working/Transolver_plus/output/checkpoint_best.pth \\
        --fold_id 0 --gpu 0
"""

import sys, os, argparse
import torch
import numpy as np
from torch_geometric.data import Data

from models.Transolver_plus import Model

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',   required=True,
                    help='Preprocessed dir containing param0/…/x.npy etc.')
parser.add_argument('--checkpoint', required=True,
                    help='Path to checkpoint_best.pth or checkpoint_latest.pth')
parser.add_argument('--fold_id',    default=0, type=int,
                    help='Validation fold index (0-8, must match training)')
parser.add_argument('--gpu',        default=0, type=int)
args = parser.parse_args()

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
print(f'[INFO] Device: {device}')

# ── Enumerate validation samples ───────────────────────────────────────────
val_fold     = f'param{args.fold_id}'
val_fold_dir = os.path.join(args.save_dir, val_fold)
if not os.path.isdir(val_fold_dir):
    raise FileNotFoundError(
        f'Validation fold directory not found: {val_fold_dir}\n'
        f'Check --save_dir points to the preprocessed root with param0/, param1/, …'
    )

val_samples = sorted([
    os.path.join(val_fold, d)
    for d in os.listdir(val_fold_dir)
    if os.path.isfile(os.path.join(val_fold_dir, d, 'x.npy'))
])
if not val_samples:
    raise RuntimeError(f'No preprocessed samples found under {val_fold_dir}')
print(f'[INFO] Validation fold: {val_fold}  ({len(val_samples)} samples)')

# ── Load a single sample from .npy files ──────────────────────────────────
def load_sample(save_dir, rel_path):
    p = os.path.join(save_dir, rel_path)
    x          = torch.tensor(np.load(os.path.join(p, 'x.npy')),          dtype=torch.float)
    y          = torch.tensor(np.load(os.path.join(p, 'y.npy')),          dtype=torch.float)
    pos        = torch.tensor(np.load(os.path.join(p, 'pos.npy')),        dtype=torch.float)
    surf       = torch.tensor(np.load(os.path.join(p, 'surf.npy')),       dtype=torch.bool)
    return Data(x=x, y=y, pos=pos, surf=surf)

print('[INFO] Loading validation samples …')
val_data = [load_sample(args.save_dir, rel) for rel in val_samples]
print(f'[INFO] Loaded {len(val_data)} samples.')

# ── Rebuild model (same hyperparams as main_car.py) ───────────────────────
model = Model(
    n_hidden=256, n_layers=4, space_dim=7,
    fun_dim=0, n_head=8, mlp_ratio=2,
    out_dim=4, slice_num=32, unified_pos=0, dropout=0.1,
)

ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()
print(f'[INFO] Loaded checkpoint — epoch {ckpt["epoch"]}  '
      f'val_l2re (training metric) = {ckpt["val_loss_l2re"]:.6f}')

# ── Evaluation loop ────────────────────────────────────────────────────────
# Both model output and data.y are in NORMALIZED space — compare directly.
# surf_mask selects surface pressure nodes; ~surf_mask selects volume velocity nodes.

l2re_surf_list = []   # Surf  metric
l2re_vol_list  = []   # Volume metric
l2re_train_list = []  # all-node last-channel (matches training val metric)

with torch.no_grad():
    for idx, data in enumerate(val_data):
        x         = data.x.unsqueeze(0).to(device)        # (1, N, 7)
        pos       = data.pos.unsqueeze(0).to(device)       # (1, N, 3)
        condition = torch.zeros(1, 3, device=device)       # (1, 3)
        y         = data.y.to(device)                      # (N, 4) — normalized
        surf_mask = data.surf.to(device)                   # (N,) bool

        out = model((x, pos, condition)).squeeze(0)        # (N, 4) — normalized

        # Surf: surface pressure (last channel)
        pred_p = out[surf_mask, -1]
        gt_p   = y[surf_mask, -1]
        l2re_s = (torch.norm(pred_p - gt_p) / (torch.norm(gt_p) + 1e-8)).item()
        l2re_surf_list.append(l2re_s)

        # Volume: surrounding velocity (first 3 channels)
        pred_v = out[~surf_mask, :3]
        gt_v   = y[~surf_mask, :3]
        l2re_v = (torch.norm(pred_v - gt_v) / (torch.norm(gt_v) + 1e-8)).item()
        l2re_vol_list.append(l2re_v)

        # Training-style metric: all nodes, last channel only
        l2re_t = (torch.norm(out[:, -1] - y[:, -1]) / (torch.norm(y[:, -1]) + 1e-8)).item()
        l2re_train_list.append(l2re_t)

        if (idx + 1) % 10 == 0 or idx == len(val_data) - 1:
            print(f'  [{idx+1:>3}/{len(val_data)}]  '
                  f'surf={l2re_s:.4f}  vol={l2re_v:.4f}  train_metric={l2re_t:.4f}')

# ── Aggregate results ──────────────────────────────────────────────────────
mean_surf  = np.mean(l2re_surf_list)
mean_vol   = np.mean(l2re_vol_list)
mean_train = np.mean(l2re_train_list)

print()
print('=' * 60)
print('EVALUATION RESULTS — Transolver+ ShapeNet Car')
print('(all metrics in normalized space)')
print('=' * 60)
print(f'  Surf   L2RE (surface pressure)    : {mean_surf:.4f}')
print(f'  Volume L2RE (surrounding velocity): {mean_vol:.4f}')
print(f'  Training val metric (all-node pressure L2RE): {mean_train:.4f}')
print()
print('Paper Table 3 (for reference — absolute values may differ if paper')
print('uses de-normalized space; use this as a relative comparison):')
print('  Transolver baseline: Volume=0.0207, Surf=0.0745')
print('  3D-GeoCA           : Volume=0.0319, Surf=0.0779')
print('  GNOT               : Volume=0.0329, Surf=0.0798')
print('=' * 60)
print()
print('NOTE: CD / rho_D skipped — requires raw .vtk files not present on Kaggle.')
print(f'      Best val_l2re from training: {ckpt["val_loss_l2re"]:.4f}')
