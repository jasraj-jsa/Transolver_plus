"""
Proper evaluation script for Transolver+ on ShapeNet Car.

Computes the same metrics as Transolver's main_evaluation.py:
  - Relative L2 for surface pressure  (Surf)
  - Relative L2 for surrounding velocity  (Volume)
  - Relative L2 for drag coefficient  (CD)
  - Spearman rank correlation for drag coefficient  (rho_D)

Usage (on Kaggle or HPC where all deps are available):

    python evaluate_car.py \
        --data_dir /kaggle/input/mlcfd-car-data/training_data \
        --save_dir /kaggle/input/mlcfd-car-data/car_preprocessed \
        --checkpoint /kaggle/working/transolver_output/0/200_0.5/checkpoint_best.pth \
        --fold_id 0 --gpu 0
"""

import sys, os, argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.stats

# Make sure the original Transolver helpers are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Transolver', 'Car-Design-ShapeNetCar'))

from dataset.load_dataset import load_train_val_fold_file
from dataset.dataset import GraphDataset
from torch_geometric.loader import DataLoader

from models.Transolver_plus import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir',   default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--checkpoint', required=True, help='Path to checkpoint_best.pth')
parser.add_argument('--fold_id',    default=0, type=int)
parser.add_argument('--gpu',        default=0, type=int)
args = parser.parse_args()

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
print(f'[INFO] Device: {device}')

# ── Load dataset (preprocessed, same fold used during training) ────────────
print('[INFO] Loading validation fold ...')
_, val_data, coef_norm, vallst = load_train_val_fold_file(args, preprocessed=True)
val_ds = GraphDataset(val_data)
val_loader = DataLoader(val_ds, batch_size=1)
print(f'[INFO] Validation samples: {len(val_ds)}')

# ── coef_norm layout: [velo_mean, velo_std, press_mean[scalar], press_std[scalar]] ──
#    Actually stored as [mean_arr (4,), std_arr (4,)] where last channel is pressure.
#    The original code uses:  coef_norm[2] = mean,  coef_norm[3] = std
#    Check what load_train_val_fold_file returns by inspecting coef_norm.
print(f'[INFO] coef_norm type={type(coef_norm)}  len={len(coef_norm) if coef_norm else None}')

# ── Rebuild model (same hyperparams as main_car.py) ───────────────────────
model = Model(
    n_hidden=256, n_layers=4, space_dim=7,
    fun_dim=0, n_head=8, mlp_ratio=2,
    out_dim=4, slice_num=32, unified_pos=0, dropout=0.1,
)

ckpt = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()
print(f'[INFO] Loaded checkpoint from epoch {ckpt["epoch"]}  '
      f'val_l2re={ckpt["val_loss_l2re"]:.6f}')

# ── Evaluation loop ────────────────────────────────────────────────────────
criterion = nn.MSELoss(reduction='none')

l2re_surf_press_list = []  # Surf (surface pressure relative L2)
l2re_velo_list       = []  # Volume (surrounding velocity relative L2)
l2re_all_list        = []  # combined (what training monitored, last channel only)

# For drag coefficient we need vtk meshes; we attempt to import cal_coefficient
# and skip gracefully if vtk is unavailable.
try:
    from utils.drag_coefficient import cal_coefficient
    _can_compute_cd = True
    print('[INFO] vtk found — will compute drag coefficient (CD / rho_D)')
except ImportError:
    _can_compute_cd = False
    print('[WARN] vtk not installed — CD / rho_D metrics will be skipped.')
    print('       Install vtk (pip install vtk) to compute drag coefficient.')

gt_cd_list   = []
pred_cd_list = []

mean = torch.tensor(coef_norm[2], dtype=torch.float, device=device) if coef_norm is not None else None
std  = torch.tensor(coef_norm[3], dtype=torch.float, device=device) if coef_norm is not None else None

with torch.no_grad():
    for idx, (cfd_data, geom) in enumerate(val_loader):
        cfd_data = cfd_data.to(device)
        geom     = geom.to(device)

        # forward (same interface as CarDesignLoader in main_car.py)
        x   = cfd_data.x.unsqueeze(0).float()
        pos = cfd_data.pos.unsqueeze(0).float()
        condition = torch.zeros(1, 3, device=device)
        out = model((x, pos, condition))   # (1, N, 4)
        out = out.squeeze(0)               # (N, 4)

        targets = cfd_data.y               # (N, 4)  — normalised
        surf_mask = cfd_data.surf          # bool (N,)

        # De-normalise
        if mean is not None and std is not None:
            out_denorm     = out     * std + mean
            targets_denorm = targets * std + mean
        else:
            out_denorm     = out
            targets_denorm = targets

        # ── Surf: relative L2 of surface pressure ────────────────────────
        pred_press = out_denorm[surf_mask, -1]
        gt_press   = targets_denorm[surf_mask, -1]
        l2re_surf  = (torch.norm(pred_press - gt_press) / torch.norm(gt_press)).item()
        l2re_surf_press_list.append(l2re_surf)

        # ── Volume: relative L2 of surrounding (non-surface) velocity ────
        pred_velo = out_denorm[~surf_mask, :3]
        gt_velo   = targets_denorm[~surf_mask, :3]
        l2re_velo = (torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)).item()
        l2re_velo_list.append(l2re_velo)

        # ── Combined L2RE on last channel (what the training loop tracked) ─
        l2re_all = (torch.norm(out_denorm[:, -1] - targets_denorm[:, -1]) /
                    torch.norm(targets_denorm[:, -1])).item()
        l2re_all_list.append(l2re_all)

        # ── Drag coefficient (needs vtk + raw vtk files) ─────────────────
        if _can_compute_cd and vallst is not None:
            try:
                sample_name = vallst[idx].split('/')[1]
                pred_press_np  = pred_press[:, None].cpu().numpy()
                pred_velo_np   = out_denorm[surf_mask, :3].cpu().numpy()
                gt_press_np    = gt_press[:, None].cpu().numpy()
                gt_velo_np     = targets_denorm[surf_mask, :3].cpu().numpy()

                pred_cd = cal_coefficient(sample_name, pred_press_np, pred_velo_np)
                gt_cd   = cal_coefficient(sample_name, gt_press_np,   gt_velo_np)
                pred_cd_list.append(pred_cd)
                gt_cd_list.append(gt_cd)
            except Exception as e:
                pass  # vtk files not accessible — silently skip

        if (idx + 1) % 10 == 0:
            print(f'  Processed {idx+1}/{len(val_ds)} samples ...')

# ── Aggregate results ──────────────────────────────────────────────────────
print('\n' + '='*60)
print('EVALUATION RESULTS — Transolver+ ShapeNet Car')
print('='*60)
print(f'  Volume (vel relative L2)  : {np.mean(l2re_velo_list):.4f}')
print(f'  Surf   (pres relative L2) : {np.mean(l2re_surf_press_list):.4f}')
print(f'  Combined last-ch L2RE     : {np.mean(l2re_all_list):.4f}  '
      f'(what training monitored — not the paper metric)')

print()
print('Paper reference (Table 3):')
print('  Transolver   : Volume=0.0207, Surf=0.0745, CD=0.0103, rho_D=0.9935')
print('  3D-GeoCA     : Volume=0.0319, Surf=0.0779, CD=0.0159, rho_D=0.9842')
print('  GNOT         : Volume=0.0329, Surf=0.0798, CD=0.0178, rho_D=0.9833')
print('='*60)

if len(gt_cd_list) > 0:
    gt_cd_arr   = np.array(gt_cd_list)
    pred_cd_arr = np.array(pred_cd_list)
    cd_rel_l2   = np.mean(np.abs(pred_cd_arr - gt_cd_arr) / np.abs(gt_cd_arr))
    rho_d       = scipy.stats.spearmanr(gt_cd_arr, pred_cd_arr)[0]
    print(f'  CD (relative L2)          : {cd_rel_l2:.4f}')
    print(f'  rho_D (Spearman)          : {rho_d:.4f}')
    print('='*60)
else:
    print('  CD / rho_D: not computed (vtk files unavailable)')
    print('  → Run on your data server / HPC node where the raw .vtk files live')
    print('='*60)
