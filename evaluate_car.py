"""
Evaluation script for Transolver+ on ShapeNet Car (preprocessed .npy data).

The .npy files store RAW (un-normalized) x and y values.
Training computed coef_norm over the training folds and normalized y in memory.
This script replicates that: computes coef_norm from training folds, normalizes
val y the same way, then compares model output (also in normalized space) against
the normalized targets.

Metrics:
  - Surf   : relative L2 on surface pressure nodes    (last channel, surf==True)
  - Volume : relative L2 on surrounding velocity nodes (first 3 channels, surf==False)

CD / rho_D skipped (requires raw .vtk files not present on Kaggle).

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
                    help='Preprocessed dir containing param0/…/x.npy etc. (raw values)')
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

# ── Enumerate all samples in each fold ────────────────────────────────────
def list_fold_samples(save_dir, fold_id):
    fold_dir = os.path.join(save_dir, f'param{fold_id}')
    if not os.path.isdir(fold_dir):
        return []
    return sorted([
        os.path.join(f'param{fold_id}', d)
        for d in os.listdir(fold_dir)
        if os.path.isfile(os.path.join(fold_dir, d, 'x.npy'))
    ])

val_samples = list_fold_samples(args.save_dir, args.fold_id)
if not val_samples:
    raise RuntimeError(f'No val samples in param{args.fold_id} under {args.save_dir}')
print(f'[INFO] Validation fold: param{args.fold_id}  ({len(val_samples)} samples)')

train_samples = []
for fi in range(9):
    if fi == args.fold_id:
        continue
    train_samples += list_fold_samples(args.save_dir, fi)
print(f'[INFO] Training samples found: {len(train_samples)}')

if not train_samples:
    raise RuntimeError('No training samples found — cannot compute coef_norm.')

# ── Load raw y arrays from training folds and compute mean/std ────────────
# Matches get_datalist(norm=True): running mean over all nodes × samples,
# then running variance pass. Uses numpy concat to avoid the broken
# online-variance formula from the previous version.
print('[INFO] Computing coef_norm from training set …')
all_y = []
for rel in train_samples:
    p = os.path.join(args.save_dir, rel)
    all_y.append(np.load(os.path.join(p, 'y.npy')).astype(np.float64))

all_y_cat = np.concatenate(all_y, axis=0)   # (total_nodes, 4)
mean_out = all_y_cat.mean(axis=0).astype(np.float32)
std_out  = all_y_cat.std(axis=0).astype(np.float32)
del all_y, all_y_cat

print(f'[INFO] mean_out = {mean_out}')
print(f'[INFO] std_out  = {std_out}')

mean_out_t = torch.tensor(mean_out)
std_out_t  = torch.tensor(std_out)

# ── Load a single sample ───────────────────────────────────────────────────
def load_sample(save_dir, rel_path):
    p = os.path.join(save_dir, rel_path)
    x    = torch.tensor(np.load(os.path.join(p, 'x.npy')),    dtype=torch.float)
    y    = torch.tensor(np.load(os.path.join(p, 'y.npy')),    dtype=torch.float)
    pos  = torch.tensor(np.load(os.path.join(p, 'pos.npy')),  dtype=torch.float)
    surf = torch.tensor(np.load(os.path.join(p, 'surf.npy')), dtype=torch.bool)
    # Normalize x and y exactly as get_datalist does (coef_norm path)
    # NOTE: training also normalizes x with mean_in/std_in, but since main_car.py
    # passes out_norm=0, pos_norm=0 — training used the already-normalized x from
    # get_datalist and did NOT re-normalize inside train(). So we need x normalized.
    # The coef_norm for x was computed the same way over training set.
    # For simplicity we normalize y here; x normalization was already baked into
    # the .npy files by get_datalist when savedir was set. Check: if x was saved
    # AFTER normalization or BEFORE.
    # From dataset.py line 206-210: save happens BEFORE the norm loop (line 249).
    # So .npy x is RAW too. But main_car.py's train() uses out_norm=0, pos_norm=0
    # meaning it does NOT normalize inside the train loop. The normalization of x
    # was applied by get_datalist returning normalized data.x in memory.
    # We need to replicate that for x as well.
    return Data(x=x, y=y, pos=pos, surf=surf)

# Compute mean_in / std_in for x too (same logic)
print('[INFO] Computing x normalization stats …')
all_x = []
for rel in train_samples:
    p = os.path.join(args.save_dir, rel)
    all_x.append(np.load(os.path.join(p, 'x.npy')).astype(np.float64))
all_x_cat = np.concatenate(all_x, axis=0)
mean_in = all_x_cat.mean(axis=0).astype(np.float32)
std_in  = all_x_cat.std(axis=0).astype(np.float32)
del all_x, all_x_cat
print(f'[INFO] mean_in  = {mean_in}')
print(f'[INFO] std_in   = {std_in}')

mean_in_t = torch.tensor(mean_in)
std_in_t  = torch.tensor(std_in)

print('[INFO] Loading validation samples …')
val_data = [load_sample(args.save_dir, rel) for rel in val_samples]
print(f'[INFO] Loaded {len(val_data)} samples.')

# ── Rebuild model ──────────────────────────────────────────────────────────
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
l2re_surf_list  = []
l2re_vol_list   = []
l2re_train_list = []

with torch.no_grad():
    for idx, data in enumerate(val_data):
        # Normalize x and y the same way get_datalist does in memory
        x_norm = ((data.x - mean_in_t) / (std_in_t + 1e-8)).unsqueeze(0).to(device)
        y_norm = ((data.y - mean_out_t) / (std_out_t + 1e-8)).to(device)
        pos    = data.pos.unsqueeze(0).to(device)
        condition = torch.zeros(1, 3, device=device)
        surf_mask = data.surf.to(device)

        out = model((x_norm, pos, condition)).squeeze(0)   # (N, 4) — normalized

        # Surf: surface pressure (last channel)
        pred_p = out[surf_mask, -1]
        gt_p   = y_norm[surf_mask, -1]
        l2re_s = (torch.norm(pred_p - gt_p) / (torch.norm(gt_p) + 1e-8)).item()
        l2re_surf_list.append(l2re_s)

        # Volume: surrounding velocity (first 3 channels)
        pred_v = out[~surf_mask, :3]
        gt_v   = y_norm[~surf_mask, :3]
        l2re_v = (torch.norm(pred_v - gt_v) / (torch.norm(gt_v) + 1e-8)).item()
        l2re_vol_list.append(l2re_v)

        # Training-style metric: all nodes, last channel (matches training val_l2re)
        l2re_t = (torch.norm(out[:, -1] - y_norm[:, -1]) /
                  (torch.norm(y_norm[:, -1]) + 1e-8)).item()
        l2re_train_list.append(l2re_t)

        if (idx + 1) % 10 == 0 or idx == len(val_data) - 1:
            print(f'  [{idx+1:>3}/{len(val_data)}]  '
                  f'surf={l2re_s:.4f}  vol={l2re_v:.4f}  train_metric={l2re_t:.4f}')

# ── Results ────────────────────────────────────────────────────────────────
mean_surf  = np.mean(l2re_surf_list)
mean_vol   = np.mean(l2re_vol_list)
mean_train = np.mean(l2re_train_list)

print()
print('=' * 60)
print('EVALUATION RESULTS — Transolver+ ShapeNet Car')
print('=' * 60)
print(f'  Surf   L2RE (surface pressure)    : {mean_surf:.4f}')
print(f'  Volume L2RE (surrounding velocity): {mean_vol:.4f}')
print(f'  Training val metric (all-node, last-channel L2RE): {mean_train:.4f}')
print(f'  [Expected ~{ckpt["val_loss_l2re"]:.4f} from checkpoint]')
print()
print('Paper Table 3 (Transolver baseline):')
print('  Volume=0.0207  Surf=0.0745  CD=0.0103  rho_D=0.9935')
print('=' * 60)
print()
print('NOTE: CD / rho_D skipped — requires raw .vtk files not on Kaggle.')
