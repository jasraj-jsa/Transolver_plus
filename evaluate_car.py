"""
evaluate_car.py — Final evaluation of Transolver+ on ShapeNet Car.

Computes the paper metrics (Table 3):
  - Surf   L2RE : relative L2 on surface pressure nodes  (surf==True,  last channel)
  - Volume L2RE : relative L2 on surrounding velocity    (surf==False, first 3 channels)

The .npy files store RAW (un-normalized) data. Normalization (coef_norm) is recomputed
from the training folds, exactly as load_train_val_fold does during training.

Usage:
    python evaluate_car.py \
        --save_dir ./dataset/mlcfd_data/car_preprocessed \
        --checkpoint ./output/0/500_0.5/checkpoint_best.pth \
        --fold_id 0 --gpu 0
"""

import sys, os, argparse, json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

from models.Transolver_plus import Model

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',   required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--fold_id',    default=0, type=int)
parser.add_argument('--gpu',        default=0, type=int)
args = parser.parse_args()

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
print(f'[INFO] Device: {device}')

# ── Enumerate samples ──────────────────────────────────────────────────────
def list_fold(save_dir, fold_id):
    d = os.path.join(save_dir, f'param{fold_id}')
    if not os.path.isdir(d):
        return []
    return sorted([
        os.path.join(f'param{fold_id}', s)
        for s in os.listdir(d)
        if os.path.isfile(os.path.join(d, s, 'x.npy'))
    ])

val_samples   = list_fold(args.save_dir, args.fold_id)
train_samples = []
for fi in range(9):
    if fi != args.fold_id:
        train_samples += list_fold(args.save_dir, fi)

print(f'[INFO] Val samples: {len(val_samples)}  Train samples: {len(train_samples)}')
if not val_samples or not train_samples:
    raise RuntimeError('Missing samples — check --save_dir')

# ── Compute coef_norm from training folds (raw .npy → same as load_train_val_fold) ──
print('[INFO] Computing coef_norm from training set …')
all_x, all_y = [], []
for rel in train_samples:
    p = os.path.join(args.save_dir, rel)
    all_x.append(np.load(os.path.join(p, 'x.npy')).astype(np.float64))
    all_y.append(np.load(os.path.join(p, 'y.npy')).astype(np.float64))

cat_x = np.concatenate(all_x, axis=0)
cat_y = np.concatenate(all_y, axis=0)
mean_in  = cat_x.mean(axis=0).astype(np.float32)
std_in   = cat_x.std(axis=0).astype(np.float32)
mean_out = cat_y.mean(axis=0).astype(np.float32)
std_out  = cat_y.std(axis=0).astype(np.float32)
del all_x, all_y, cat_x, cat_y

mean_in_t  = torch.tensor(mean_in)
std_in_t   = torch.tensor(std_in)
mean_out_t = torch.tensor(mean_out)
std_out_t  = torch.tensor(std_out)
print(f'[INFO] mean_out={mean_out}  std_out={std_out}')

# ── Load val samples (raw, normalize in memory) ────────────────────────────
def load_sample(rel):
    p = os.path.join(args.save_dir, rel)
    x    = torch.tensor(np.load(os.path.join(p, 'x.npy')),    dtype=torch.float)
    y    = torch.tensor(np.load(os.path.join(p, 'y.npy')),    dtype=torch.float)
    pos  = torch.tensor(np.load(os.path.join(p, 'pos.npy')),  dtype=torch.float)
    surf = torch.tensor(np.load(os.path.join(p, 'surf.npy')), dtype=torch.bool)
    x_n  = (x - mean_in_t)  / (std_in_t  + 1e-8)
    y_n  = (y - mean_out_t) / (std_out_t + 1e-8)
    return Data(x=x_n, y=y_n, pos=pos, surf=surf)

print('[INFO] Loading val samples …')
val_data = [load_sample(r) for r in val_samples]

# ── Model ──────────────────────────────────────────────────────────────────
model = Model(
    n_hidden=256, n_layers=4, space_dim=7,
    fun_dim=0, n_head=8, mlp_ratio=2,
    out_dim=4, slice_num=32, unified_pos=0, dropout=0.1,
)
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()
epoch = ckpt.get('epoch', '?')
print(f'[INFO] Loaded checkpoint — epoch {epoch}')

# ── Evaluation ─────────────────────────────────────────────────────────────
criterion = nn.MSELoss(reduction='none')

surf_l2re_list = []
vol_l2re_list  = []
press_mse_list = []
velo_mse_list  = []

with torch.no_grad():
    for idx, data in enumerate(val_data):
        x         = data.x.unsqueeze(0).to(device)
        pos       = data.pos.unsqueeze(0).to(device)
        y         = data.y.to(device)
        surf      = data.surf.to(device)
        condition = torch.zeros(1, 3, device=device)

        out = model((x, pos, condition)).squeeze(0)   # (N, 4)

        # Paper metrics (normalized space — same normalization both sides)
        surf_l2re = (torch.norm(out[surf,  -1] - y[surf,  -1]) /
                     (torch.norm(y[surf,  -1]) + 1e-8)).item()
        vol_l2re  = (torch.norm(out[~surf, :3] - y[~surf, :3]) /
                     (torch.norm(y[~surf, :3]) + 1e-8)).item()
        surf_l2re_list.append(surf_l2re)
        vol_l2re_list.append(vol_l2re)

        # Training-style MSE losses (for cross-check with training log)
        press_mse = criterion(out[surf,  -1], y[surf,  -1]).mean().item()
        velo_mse  = criterion(out[:, :3],     y[:, :3]).mean().item()
        press_mse_list.append(press_mse)
        velo_mse_list.append(velo_mse)

        if (idx + 1) % 10 == 0 or idx == len(val_data) - 1:
            print(f'  [{idx+1:>3}/{len(val_data)}]  '
                  f'surf_l2re={surf_l2re:.4f}  vol_l2re={vol_l2re:.4f}')

# ── Results ────────────────────────────────────────────────────────────────
mean_surf = np.mean(surf_l2re_list)
mean_vol  = np.mean(vol_l2re_list)
mean_press_mse = np.mean(press_mse_list)
mean_velo_mse  = np.mean(velo_mse_list)
val_loss = mean_velo_mse + 0.5 * mean_press_mse

print()
print('=' * 58)
print('EVALUATION RESULTS — Transolver+ ShapeNet Car')
print('=' * 58)
print(f'  Surf   L2RE (surface pressure)  : {mean_surf:.4f}')
print(f'  Volume L2RE (surrounding vel.)  : {mean_vol:.4f}')
print(f'  Val loss (velo + 0.5*press MSE) : {val_loss:.6f}')
print()
print('  Paper Table 3 (Transolver baseline):')
print('    Surf L2RE   = 0.0745')
print('    Volume L2RE = 0.0207')
print()
delta_surf = mean_surf - 0.0745
delta_vol  = mean_vol  - 0.0207
print(f'  Delta vs baseline:')
print(f'    Surf   : {delta_surf:+.4f}  '
      f'({"better" if delta_surf < 0 else "worse"})')
print(f'    Volume : {delta_vol:+.4f}  '
      f'({"better" if delta_vol  < 0 else "worse"})')
print('=' * 58)
print()
print('NOTE: CD / rho_D require raw .vtk files — skipped.')

# Save results next to checkpoint
out_dir = os.path.dirname(args.checkpoint)
result = {
    'checkpoint_epoch': epoch,
    'surf_l2re':   mean_surf,
    'vol_l2re':    mean_vol,
    'val_loss':    val_loss,
    'paper_baseline': {'surf_l2re': 0.0745, 'vol_l2re': 0.0207},
    'per_sample': {
        'surf_l2re': surf_l2re_list,
        'vol_l2re':  vol_l2re_list,
    }
}
result_path = os.path.join(out_dir, 'eval_results.json')
with open(result_path, 'w') as f:
    json.dump(result, f, indent=4)
print(f'Saved {result_path}')
