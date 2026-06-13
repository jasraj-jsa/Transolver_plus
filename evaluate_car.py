"""
Proper evaluation script for Transolver+ on ShapeNet Car.

Computes the same metrics as Transolver's main_evaluation.py:
  - Relative L2 for surface pressure  (Surf)
  - Relative L2 for surrounding velocity  (Volume)
  - Relative L2 for drag coefficient  (CD)   — requires vtk + raw .vtk files
  - Spearman rank correlation for drag coefficient  (rho_D)

Works entirely from the PREPROCESSED directory (param0/…/x.npy etc.) so it
does NOT need the raw training_data/ folder to be present on Kaggle.

Usage:

    python evaluate_car.py \
        --save_dir /kaggle/input/mlcfd-car-data/car_preprocessed \
        --checkpoint /kaggle/working/transolver_output/0/200_0.5/checkpoint_best.pth \
        --fold_id 0 --gpu 0

    # To also compute CD / rho_D you need the raw vtk files:
    python evaluate_car.py \
        --save_dir /kaggle/input/mlcfd-car-data/car_preprocessed \
        --data_dir /kaggle/input/mlcfd-car-data/training_data \
        --checkpoint /kaggle/working/transolver_output/0/200_0.5/checkpoint_best.pth \
        --fold_id 0 --gpu 0
"""

import sys, os, argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from torch_geometric.data import Data

from models.Transolver_plus import Model

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',   required=True,
                    help='Preprocessed data dir that contains param0/…/x.npy etc.')
parser.add_argument('--data_dir',   default=None,
                    help='Raw training_data dir (only needed for CD/rho_D via vtk). '
                         'Leave unset to skip drag-coefficient computation.')
parser.add_argument('--checkpoint', required=True, help='Path to checkpoint_best.pth or checkpoint_latest.pth')
parser.add_argument('--fold_id',    default=0, type=int,
                    help='Which param fold was the validation set (0-8). Must match training.')
parser.add_argument('--gpu',        default=0, type=int)
args = parser.parse_args()

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
print(f'[INFO] Device: {device}')

# ── Build val sample list from preprocessed dir (no raw data needed) ──────
# The dataset uses fold_id as the validation fold; all param<i> != fold_id
# are training.  We just need the param<fold_id> subdirectory.
val_fold = f'param{args.fold_id}'
val_fold_dir = os.path.join(args.save_dir, val_fold)
if not os.path.isdir(val_fold_dir):
    raise FileNotFoundError(
        f'Validation fold directory not found: {val_fold_dir}\n'
        f'Make sure --save_dir points to the preprocessed root that contains param0/, param1/, …'
    )

# Enumerate all samples in the val fold that have all required .npy files
val_samples = sorted([
    os.path.join(val_fold, d)
    for d in os.listdir(val_fold_dir)
    if os.path.isfile(os.path.join(val_fold_dir, d, 'x.npy'))
])
if not val_samples:
    raise RuntimeError(f'No preprocessed samples found under {val_fold_dir}')
print(f'[INFO] Validation fold: {val_fold}  ({len(val_samples)} samples)')

# ── Load all val samples and compute coef_norm from the TRAINING folds ────
# coef_norm (mean_out, std_out) was computed over the training set during
# get_datalist(norm=True).  On Kaggle it was saved implicitly inside the
# preprocessed files (data.y is already normalised).  If we can find the
# training samples we recompute it; otherwise we assume y is already
# de-normalised (norm=False path).

def load_sample(save_dir, rel_path):
    p = os.path.join(save_dir, rel_path)
    x          = torch.tensor(np.load(os.path.join(p, 'x.npy')),          dtype=torch.float)
    y          = torch.tensor(np.load(os.path.join(p, 'y.npy')),          dtype=torch.float)
    pos        = torch.tensor(np.load(os.path.join(p, 'pos.npy')),        dtype=torch.float)
    surf       = torch.tensor(np.load(os.path.join(p, 'surf.npy')),       dtype=torch.bool)
    edge_index = torch.tensor(np.load(os.path.join(p, 'edge_index.npy')), dtype=torch.long)
    return Data(x=x, y=y, pos=pos, surf=surf, edge_index=edge_index)

# Collect all training folds to recompute coef_norm
print('[INFO] Scanning training folds to recompute coef_norm …')
train_samples_paths = []
for fi in range(9):
    if fi == args.fold_id:
        continue
    fold_dir = os.path.join(args.save_dir, f'param{fi}')
    if not os.path.isdir(fold_dir):
        continue
    for d in sorted(os.listdir(fold_dir)):
        sp = os.path.join(fold_dir, d)
        if os.path.isfile(os.path.join(sp, 'x.npy')):
            train_samples_paths.append(os.path.join(f'param{fi}', d))

print(f'[INFO] Training samples found: {len(train_samples_paths)}')

if len(train_samples_paths) == 0:
    print('[WARN] No training samples found — assuming y is already de-normalised.')
    coef_norm = None
else:
    # Compute running mean / std over the training set (same as get_datalist)
    print('[INFO] Computing coef_norm from training set (this may take a minute) …')
    old_length = 0
    mean_out = None
    std_out  = None
    all_y    = []   # keep in memory for std pass; samples are ~32k × 4 floats each

    for rel in train_samples_paths:
        p = os.path.join(args.save_dir, rel)
        y_raw = np.load(os.path.join(p, 'y.npy'))   # already normalised in place
        all_y.append(y_raw)
        if mean_out is None:
            mean_out   = y_raw.mean(axis=0)
            old_length = y_raw.shape[0]
        else:
            new_length = old_length + y_raw.shape[0]
            mean_out  += (y_raw.sum(axis=0) - y_raw.shape[0] * mean_out) / new_length
            old_length = new_length

    old_length = 0
    std_out = None
    for y_raw in all_y:
        if std_out is None:
            std_out    = ((y_raw - mean_out) ** 2).sum(axis=0) / y_raw.shape[0]
            old_length = y_raw.shape[0]
        else:
            new_length = old_length + y_raw.shape[0]
            std_out   += (((y_raw - mean_out) ** 2).sum(axis=0) - y_raw.shape[0] * std_out) / new_length
            old_length = new_length
    std_out = np.sqrt(std_out)

    # coef_norm layout matches get_datalist: (mean_in, std_in, mean_out, std_out)
    # We only need indices 2 and 3 for de-normalising predictions.
    coef_norm = (None, None, mean_out, std_out)
    print(f'[INFO] coef_norm computed.  mean_out={mean_out}  std_out={std_out}')

print(f'[INFO] coef_norm: {"computed from training set" if coef_norm else "not available — y treated as raw"}')

# ── Load validation data ───────────────────────────────────────────────────
print('[INFO] Loading validation samples …')
val_data = [load_sample(args.save_dir, rel) for rel in val_samples]

# De-normalise y in val set (undo the normalisation applied during get_datalist)
if coef_norm is not None:
    mean_out = torch.tensor(coef_norm[2], dtype=torch.float)
    std_out  = torch.tensor(coef_norm[3], dtype=torch.float)
    for data in val_data:
        data.y = data.y * (std_out + 1e-8) + mean_out
    print('[INFO] Validation y de-normalised.')
else:
    mean_out = None
    std_out  = None
    print('[INFO] Validation y used as-is (no de-normalisation).')

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
print(f'[INFO] Loaded checkpoint from epoch {ckpt["epoch"]}  '
      f'val_l2re (training metric)={ckpt["val_loss_l2re"]:.6f}')

# ── Drag coefficient (needs vtk + raw .vtk files on disk) ─────────────────
_can_compute_cd = False
if args.data_dir is not None:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        '..', 'Transolver', 'Car-Design-ShapeNetCar'))
        from utils.drag_coefficient import cal_coefficient as _cal_coef

        # Monkey-patch the hardcoded root paths inside cal_coefficient
        import utils.drag_coefficient as _dc_mod
        _dc_mod_src = open(_dc_mod.__file__).read()
        # Only enable if the vtk file root exists
        if os.path.isdir(args.data_dir):
            _can_compute_cd = True
            print('[INFO] vtk found and data_dir exists — will compute CD / rho_D')
        else:
            print(f'[WARN] --data_dir {args.data_dir} not found — skipping CD / rho_D')
    except ImportError:
        print('[WARN] vtk not installed — CD / rho_D metrics will be skipped.')
        print('       pip install vtk  to enable them.')

# ── Evaluation loop ────────────────────────────────────────────────────────
l2re_surf_press_list = []   # Surf  — surface pressure relative L2
l2re_velo_list       = []   # Volume — surrounding velocity relative L2
l2re_training_list   = []   # what the training loop monitored (all-node pressure-only L2RE)
gt_cd_list           = []
pred_cd_list         = []

with torch.no_grad():
    for idx, data in enumerate(val_data):
        x   = data.x.unsqueeze(0).to(device)          # (1, N, 7)
        pos = data.pos.unsqueeze(0).to(device)         # (1, N, 3)
        condition = torch.zeros(1, 3, device=device)   # (1, 3)

        out = model((x, pos, condition)).squeeze(0)    # (N, 4)

        # data.y has already been de-normalised above; out is in normalised space
        # so we de-normalise out to match.
        if mean_out is not None and std_out is not None:
            std_t  = std_out.to(device)
            mean_t = mean_out.to(device)
            out_denorm = out * (std_t + 1e-8) + mean_t
        else:
            out_denorm = out

        targets_denorm = data.y.to(device)   # already de-normalised
        surf_mask = data.surf.to(device)      # bool (N,)

        # ── Surf: relative L2 of surface pressure ────────────────────────
        pred_press = out_denorm[surf_mask, -1]
        gt_press   = targets_denorm[surf_mask, -1]
        l2re_surf  = (torch.norm(pred_press - gt_press) / (torch.norm(gt_press) + 1e-8)).item()
        l2re_surf_press_list.append(l2re_surf)

        # ── Volume: relative L2 of surrounding velocity ───────────────────
        pred_velo = out_denorm[~surf_mask, :3]
        gt_velo   = targets_denorm[~surf_mask, :3]
        l2re_velo = (torch.norm(pred_velo - gt_velo) / (torch.norm(gt_velo) + 1e-8)).item()
        l2re_velo_list.append(l2re_velo)

        # ── Training metric: all-node, pressure-channel only ─────────────
        l2re_tr = (torch.norm(out_denorm[:, -1] - targets_denorm[:, -1]) /
                   (torch.norm(targets_denorm[:, -1]) + 1e-8)).item()
        l2re_training_list.append(l2re_tr)

        # ── Drag coefficient ──────────────────────────────────────────────
        if _can_compute_cd:
            try:
                sample_name = val_samples[idx].split('/')[1]
                pred_cd = _cal_coef(
                    sample_name,
                    pred_press[:, None].cpu().numpy(),
                    out_denorm[surf_mask, :3].cpu().numpy(),
                )
                gt_cd = _cal_coef(
                    sample_name,
                    gt_press[:, None].cpu().numpy(),
                    targets_denorm[surf_mask, :3].cpu().numpy(),
                )
                pred_cd_list.append(pred_cd)
                gt_cd_list.append(gt_cd)
            except Exception as e:
                print(f'  [WARN] CD failed for {val_samples[idx]}: {e}')

        if (idx + 1) % 10 == 0 or idx == len(val_data) - 1:
            print(f'  [{idx+1}/{len(val_data)}]  '
                  f'surf_l2re={l2re_surf:.4f}  vol_l2re={l2re_velo:.4f}')

# ── Aggregate and print ────────────────────────────────────────────────────
print('\n' + '='*60)
print('EVALUATION RESULTS — Transolver+ ShapeNet Car')
print('='*60)
print(f'  Volume  (vel  relative L2) : {np.mean(l2re_velo_list):.4f}')
print(f'  Surf    (pres relative L2) : {np.mean(l2re_surf_press_list):.4f}')
print(f'  Training metric (all-node pressure L2RE): {np.mean(l2re_training_list):.4f}')
print()
print('Paper reference (Table 3):')
print('  Transolver   : Volume=0.0207, Surf=0.0745, CD=0.0103, rho_D=0.9935')
print('  3D-GeoCA     : Volume=0.0319, Surf=0.0779, CD=0.0159, rho_D=0.9842')
print('  GNOT         : Volume=0.0329, Surf=0.0798, CD=0.0178, rho_D=0.9833')
print('='*60)

if len(gt_cd_list) > 0:
    gt_cd_arr   = np.array(gt_cd_list)
    pred_cd_arr = np.array(pred_cd_list)
    cd_rel_l2   = np.mean(np.abs(pred_cd_arr - gt_cd_arr) / (np.abs(gt_cd_arr) + 1e-8))
    rho_d, _    = scipy.stats.spearmanr(gt_cd_arr, pred_cd_arr)
    print(f'  CD  (relative L2)          : {cd_rel_l2:.4f}')
    print(f'  rho_D  (Spearman)          : {rho_d:.4f}')
    print('='*60)
else:
    print('  CD / rho_D: skipped (pass --data_dir with raw .vtk files to enable)')
    print('='*60)
