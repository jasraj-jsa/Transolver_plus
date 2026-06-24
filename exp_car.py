"""
exp_car.py — Transolver+ on ShapeNet Car (mlcfd).

Follows the same structure as PDE-Solving-StandardBenchmark/exp_airfoil.py
so results are directly comparable in style. Hyperparameter defaults match
the original Transolver car model (main.py in Car-Design-ShapeNetCar).

Usage:
    # Train
    python exp_car.py \
        --data_dir ./dataset/mlcfd_data/car_preprocessed \
        --gpu 0 --n-layers 8 --n-hidden 256 --slice_num 32

    # Eval only (loads checkpoint)
    python exp_car.py \
        --data_dir ./dataset/mlcfd_data/car_preprocessed \
        --gpu 0 --eval 1 --save_name car_Transolver_plus
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Transolver', 'Car-Design-ShapeNetCar'))
from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
from models.Transolver_plus import Model

# ── Args (mirrors PDE-Solving-StandardBenchmark style) ────────────────────
parser = argparse.ArgumentParser('Transolver+ ShapeNet Car Benchmark')
parser.add_argument('--lr',           type=float, default=1e-3)
parser.add_argument('--epochs',       type=int,   default=200)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--n-hidden',     type=int,   default=256)
parser.add_argument('--n-layers',     type=int,   default=8,   help='8 matches original Transolver car')
parser.add_argument('--n-heads',      type=int,   default=8)
parser.add_argument('--batch-size',   type=int,   default=1)
parser.add_argument('--mlp_ratio',    type=int,   default=2)
parser.add_argument('--dropout',      type=float, default=0.0)
parser.add_argument('--slice_num',    type=int,   default=32)
parser.add_argument('--weight',       type=float, default=0.5,
                    help='Loss weight: total = loss_velo + weight * loss_press')
parser.add_argument('--fold_id',      type=int,   default=0)
parser.add_argument('--gpu',          type=str,   default='0')
parser.add_argument('--eval',         type=int,   default=0)
parser.add_argument('--preprocessed', type=int,   default=1)
parser.add_argument('--save_name',    type=str,   default='car_Transolver_plus')
parser.add_argument('--data_dir',     type=str,   default='./dataset/mlcfd_data/car_preprocessed')
args = parser.parse_args()

# Namespace shim so load_train_val_fold sees args.data_dir / args.save_dir / args.fold_id
args.save_dir = args.data_dir

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(args)

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results/' + args.save_name, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
print('Loading data …')
train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
train_ds = GraphDataset(train_data)
val_ds   = GraphDataset(val_data)
ntrain, ntest = len(train_ds), len(val_ds)
print(f'Train: {ntrain}  Val: {ntest}')

import random

class CarLoader:
    """Yields (x, y, pos, condition, surf) — same interface as main_car.py."""
    def __init__(self, ds, shuffle=False):
        self.ds = ds
        self.shuffle = shuffle
    def __len__(self):
        return len(self.ds)
    def __iter__(self):
        idx = list(range(len(self.ds)))
        if self.shuffle:
            random.shuffle(idx)
        for i in idx:
            d, _ = self.ds[i]
            yield (d.x.unsqueeze(0).float(),
                   d.y.unsqueeze(0).float(),
                   d.pos.unsqueeze(0).float(),
                   torch.zeros(1, 3),
                   d.surf)

train_loader = CarLoader(train_ds, shuffle=True)
val_loader   = CarLoader(val_ds,   shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────────
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

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Trainable Params: {total_params:,}')

# ── Loss (relative L2, matches TestLoss in standard benchmark) ─────────────
def rel_l2(pred, target):
    """Relative L2 over a batch: mean( ||pred-target|| / ||target|| )."""
    diff = (pred - target).reshape(pred.shape[0], -1)
    denom = target.reshape(target.shape[0], -1)
    return (torch.norm(diff, dim=1) / (torch.norm(denom, dim=1) + 1e-8)).mean()

criterion = nn.MSELoss(reduction='none')

# ── Eval function (paper metrics) ──────────────────────────────────────────
@torch.no_grad()
def evaluate():
    model.eval()
    surf_l2re_list, vol_l2re_list = [], []
    for x, y, pos, cond, surf in val_loader:
        x, y, pos, cond, surf = (x.to(device), y.to(device), pos.to(device),
                                  cond.to(device), surf.to(device))
        out = model((x, pos, cond)).squeeze(0)
        tgt = y.squeeze(0)
        surf_l2re_list.append(
            (torch.norm(out[surf,  -1] - tgt[surf,  -1]) /
             (torch.norm(tgt[surf,  -1]) + 1e-8)).item()
        )
        vol_l2re_list.append(
            (torch.norm(out[~surf, :3] - tgt[~surf, :3]) /
             (torch.norm(tgt[~surf, :3]) + 1e-8)).item()
        )
    return float(np.mean(surf_l2re_list)), float(np.mean(vol_l2re_list))

# ── Eval-only mode ─────────────────────────────────────────────────────────
if args.eval:
    ckpt_path = os.path.join('./checkpoints', args.save_name + '.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    surf_l2re, vol_l2re = evaluate()
    print(f'\n{"="*50}')
    print(f'EVAL  {args.save_name}')
    print(f'  Surf   L2RE : {surf_l2re:.4f}   (paper Transolver: 0.0745)')
    print(f'  Volume L2RE : {vol_l2re:.4f}   (paper Transolver: 0.0207)')
    print(f'{"="*50}')
    sys.exit(0)

# ── Training ───────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr,
    total_steps=int((ntrain // args.batch_size + 1) * args.epochs),
    final_div_factor=1000.,
)

history = []
best_val = 1e5

for ep in tqdm(range(args.epochs)):
    # ── train ──
    model.train()
    press_losses, velo_losses = [], []
    for x, y, pos, cond, surf in train_loader:
        x, y, pos, cond, surf = (x.to(device), y.to(device), pos.to(device),
                                  cond.to(device), surf.to(device))
        optimizer.zero_grad()
        out = model((x, pos, cond)).squeeze(0)
        tgt = y.squeeze(0)
        loss_press = criterion(out[surf,  -1], tgt[surf,  -1]).mean()
        loss_velo  = criterion(out[:, :3],     tgt[:, :3]).mean()
        (loss_velo + args.weight * loss_press).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        press_losses.append(loss_press.item())
        velo_losses.append(loss_velo.item())

    train_press = float(np.mean(press_losses))
    train_velo  = float(np.mean(velo_losses))
    train_loss  = train_velo + args.weight * train_press

    # ── val every 10 epochs ──
    if ep % 10 == 0 or ep == args.epochs - 1:
        surf_l2re, vol_l2re = evaluate()
        val_loss = surf_l2re + vol_l2re   # combined for checkpoint selection
        print(f'Epoch {ep:>3}  train={train_loss:.5f}  '
              f'surf_l2re={surf_l2re:.4f}  vol_l2re={vol_l2re:.4f}')
        history.append({'epoch': ep, 'train_loss': train_loss,
                        'surf_l2re': surf_l2re, 'vol_l2re': vol_l2re})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join('./checkpoints', args.save_name + '.pt'))
    else:
        print(f'Epoch {ep:>3}  train={train_loss:.5f}')

# ── Save training log ──────────────────────────────────────────────────────
log_path = os.path.join('./results', args.save_name, 'train_log.json')
with open(log_path, 'w') as f:
    json.dump(history, f, indent=2)

# ── Final eval ────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(
    os.path.join('./checkpoints', args.save_name + '.pt'),
    map_location=device, weights_only=False))
surf_l2re, vol_l2re = evaluate()

print(f'\n{"="*58}')
print(f'FINAL RESULTS  {args.save_name}')
print(f'  Params       : {total_params:,}')
print(f'  Layers       : {args.n_layers}  Hidden: {args.n_hidden}  Slices: {args.slice_num}')
print(f'  Surf   L2RE  : {surf_l2re:.4f}   (Transolver baseline: 0.0745)')
print(f'  Volume L2RE  : {vol_l2re:.4f}   (Transolver baseline: 0.0207)')
print(f'{"="*58}')

# ── Plots ─────────────────────────────────────────────────────────────────
val_epochs = [r['epoch']    for r in history]
surfs      = [r['surf_l2re'] for r in history]
vols       = [r['vol_l2re']  for r in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(val_epochs, surfs,  color='crimson',  marker='o', markersize=3, label='Transolver+')
axes[0].axhline(0.0745, color='gray', linestyle='--', label='Transolver baseline')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Relative L2')
axes[0].set_title('Surf L2RE (surface pressure)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(val_epochs, vols, color='seagreen', marker='o', markersize=3, label='Transolver+')
axes[1].axhline(0.0207, color='gray', linestyle='--', label='Transolver baseline')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Relative L2')
axes[1].set_title('Volume L2RE (surrounding velocity)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.suptitle(f'Transolver+ ShapeNet Car  '
             f'(layers={args.n_layers}, hidden={args.n_hidden}, slices={args.slice_num})',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join('./results', args.save_name, 'metrics.png'), dpi=150)
plt.close(fig)
print(f'Plot saved to results/{args.save_name}/metrics.png')

# Save final result JSON (easy to compare across runs)
with open(os.path.join('./results', args.save_name, 'result.json'), 'w') as f:
    json.dump({
        'model': 'Transolver+',
        'task': 'ShapeNet Car',
        'n_layers': args.n_layers, 'n_hidden': args.n_hidden,
        'n_heads': args.n_heads, 'slice_num': args.slice_num,
        'nb_params': total_params, 'epochs': args.epochs,
        'surf_l2re': surf_l2re, 'vol_l2re': vol_l2re,
        'paper_baseline': {'model': 'Transolver', 'surf_l2re': 0.0745, 'vol_l2re': 0.0207},
    }, f, indent=4)
