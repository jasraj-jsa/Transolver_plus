"""
main_car.py — Runs Transolver+ on the ShapeNet Car (mlcfd) dataset.

This replicates what Car-Design-ShapeNetCar/train.py does for the original
Transolver, but using the Transolver+ model (Gumbel-softmax slice assignment).

Key design decisions (matching the original car training loop):
  - Loss = loss_velo + reg * loss_press
      loss_press: MSE on surface pressure nodes only (cfd_data.surf == True, last channel)
      loss_velo:  MSE on velocity channels of ALL nodes (first 3 channels)
  - Data normalization: handled by load_train_val_fold (coef_norm applied in memory)
  - No pos_norm or out_norm inside the training loop (data already normalized)
  - condition = zeros(1,3): car dataset has no separate flow-condition vector
"""

import sys
import os
import json
import time
import logging
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Transolver', 'Car-Design-ShapeNetCar'))

from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
from models.Transolver_plus import Model

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir',   default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id',    default=0, type=int)
parser.add_argument('--gpu',        default=0, type=int)
parser.add_argument('--val_iter',   default=10, type=int)
parser.add_argument('--weight',     default=0.5, type=float,
                    help='reg weight: total_loss = loss_velo + weight * loss_press')
parser.add_argument('--lr',         default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs',  default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--resume',     default=0, type=int)
parser.add_argument('--checkpoint_every', default=10, type=int)
parser.add_argument('--output_dir', default='', help='Override output dir (for Kaggle /working/)')
args = parser.parse_args()

# ── Device ─────────────────────────────────────────────────────────────────
n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
if use_cuda:
    torch.cuda.set_device(args.gpu)
print(f'[INFO] Device: {device}')

# ── Output dir ─────────────────────────────────────────────────────────────
base_path = args.output_dir if args.output_dir else 'metrics/car_design'
path = os.path.join(base_path, str(args.fold_id), f'{args.nb_epochs}_{args.weight}')
os.makedirs(path, exist_ok=True)
print(f'[INFO] Output dir: {path}')

log_file = os.path.join(path, 'train.log')
logging.basicConfig(filename=log_file, level=logging.INFO, filemode='a',
                    format='%(asctime)s - %(message)s')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger('').addHandler(console)
logging.info('=' * 60)
logging.info(f'[START] args: {args}')

# ── Data ───────────────────────────────────────────────────────────────────
print('[INFO] Loading dataset …')
train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
train_ds = GraphDataset(train_data)
val_ds   = GraphDataset(val_data)
print(f'[INFO] Train: {len(train_ds)}  Val: {len(val_ds)}')


class CarDesignLoader:
    """Yields (x, y, pos, condition, surf_mask) matching Transolver+ forward signature."""
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            data, _ = self.dataset[i]
            x         = data.x.unsqueeze(0).float()     # (1, N, 7)
            y         = data.y.unsqueeze(0).float()      # (1, N, 4)
            pos       = data.pos.unsqueeze(0).float()    # (1, N, 3)
            condition = torch.zeros(1, 3)                # no flow condition for car
            surf      = data.surf                        # (N,) bool
            yield x, y, pos, condition, surf


train_loader = CarDesignLoader(train_ds, shuffle=True)
val_loader   = CarDesignLoader(val_ds,   shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────────
model = Model(
    n_hidden=256, n_layers=4, space_dim=7,
    fun_dim=0, n_head=8, mlp_ratio=2,
    out_dim=4, slice_num=32, unified_pos=0, dropout=0.1,
)
model = model.to(device)
nb_params = sum(p.numel() for p in model.parameters())
logging.info(f'[INFO] Parameters: {nb_params:,}')

# ── Optimizer / scheduler ──────────────────────────────────────────────────
hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    total_steps=int((len(train_loader) // args.batch_size + 1) * args.nb_epochs),
    final_div_factor=1000.,
)

# ── Checkpoint helpers ─────────────────────────────────────────────────────
def save_ckpt(tag, epoch, train_press, train_velo, val_press, val_velo):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'train_press': train_press, 'train_velo': train_velo,
        'val_press': val_press,     'val_velo': val_velo,
    }
    torch.save(ckpt, os.path.join(path, f'checkpoint_{tag}.pth'))

def load_ckpt(tag):
    p = os.path.join(path, f'checkpoint_{tag}.pth')
    if not os.path.exists(p):
        return -1, 1e5, 1e5, 1e5, 1e5
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return (ckpt['epoch'], ckpt.get('train_press', 1e5), ckpt.get('train_velo', 1e5),
            ckpt.get('val_press', 1e5), ckpt.get('val_velo', 1e5))

start_epoch = 0
best_val_loss = 1e5
train_press = train_velo = val_press = val_velo = 1e5

if args.resume:
    start_epoch, train_press, train_velo, val_press, val_velo = load_ckpt('latest')
    if start_epoch >= 0:
        start_epoch += 1
        best_val_loss = val_press + args.weight * val_velo
        logging.info(f'[RESUME] Epoch {start_epoch}')

# ── Training loop (mirrors original Car-Design-ShapeNetCar/train.py) ───────
criterion = nn.MSELoss(reduction='none')

def train_epoch():
    model.train()
    press_losses, velo_losses = [], []
    for x, y, pos, condition, surf in train_loader:
        x         = x.to(device)
        pos       = pos.to(device)
        y         = y.to(device)
        condition = condition.to(device)
        surf      = surf.to(device)          # (N,) bool

        optimizer.zero_grad()
        out = model((x, pos, condition))     # (1, N, 4)
        out = out.squeeze(0)                 # (N, 4)
        tgt = y.squeeze(0)                   # (N, 4)

        # Surface pressure loss (surf nodes, last channel) — matches original car train.py
        loss_press = criterion(out[surf, -1], tgt[surf, -1]).mean()
        # Velocity loss (all nodes, first 3 channels) — matches original car train.py
        loss_velo  = criterion(out[:, :3],    tgt[:, :3]).mean()

        total_loss = loss_velo + args.weight * loss_press
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        press_losses.append(loss_press.item())
        velo_losses.append(loss_velo.item())

    return float(np.mean(press_losses)), float(np.mean(velo_losses))


@torch.no_grad()
def val_epoch():
    model.eval()
    press_losses, velo_losses = [], []
    surf_l2re_list, vol_l2re_list = [], []
    for x, y, pos, condition, surf in val_loader:
        x         = x.to(device)
        pos       = pos.to(device)
        y         = y.to(device)
        condition = condition.to(device)
        surf      = surf.to(device)

        out = model((x, pos, condition)).squeeze(0)
        tgt = y.squeeze(0)

        loss_press = criterion(out[surf, -1], tgt[surf, -1]).mean()
        loss_velo  = criterion(out[:, :3],    tgt[:, :3]).mean()
        press_losses.append(loss_press.item())
        velo_losses.append(loss_velo.item())

        # Paper-style L2RE metrics
        surf_l2re = (torch.norm(out[surf, -1] - tgt[surf, -1]) /
                     (torch.norm(tgt[surf, -1]) + 1e-8)).item()
        vol_l2re  = (torch.norm(out[~surf, :3] - tgt[~surf, :3]) /
                     (torch.norm(tgt[~surf, :3]) + 1e-8)).item()
        surf_l2re_list.append(surf_l2re)
        vol_l2re_list.append(vol_l2re)

    return (float(np.mean(press_losses)), float(np.mean(velo_losses)),
            float(np.mean(surf_l2re_list)), float(np.mean(vol_l2re_list)))


logging.info(f'[INFO] Training {start_epoch} → {args.nb_epochs - 1}')
pbar = tqdm(range(start_epoch, args.nb_epochs), initial=start_epoch, total=args.nb_epochs)
start_time = time.time()

for epoch in pbar:
    epoch_start = time.time()
    train_press, train_velo = train_epoch()
    train_loss = train_velo + args.weight * train_press
    epoch_time = time.time() - epoch_start

    do_val = (epoch == args.nb_epochs - 1 or epoch % args.val_iter == 0)
    if do_val:
        val_press, val_velo, surf_l2re, vol_l2re = val_epoch()
        val_loss = val_velo + args.weight * val_press
        pbar.set_postfix(train=f'{train_loss:.4f}', val=f'{val_loss:.4f}',
                         surf=f'{surf_l2re:.4f}', vol=f'{vol_l2re:.4f}')
        msg = (f'Epoch {epoch}/{args.nb_epochs-1}  '
               f'train={train_loss:.6f} (press={train_press:.4f} velo={train_velo:.4f})  '
               f'val={val_loss:.6f}  surf_l2re={surf_l2re:.4f}  vol_l2re={vol_l2re:.4f}  '
               f't={epoch_time:.1f}s')
    else:
        val_loss = val_press + args.weight * val_velo   # keep last known
        pbar.set_postfix(train=f'{train_loss:.4f}')
        msg = f'Epoch {epoch}/{args.nb_epochs-1}  train={train_loss:.6f}  t={epoch_time:.1f}s'

    logging.info(msg)

    # JSONL log
    record = {
        'epoch': epoch, 'train_loss': train_loss,
        'train_press': train_press, 'train_velo': train_velo,
        'val_loss': val_loss if do_val else None,
        'val_press': val_press if do_val else None,
        'val_velo': val_velo if do_val else None,
        'surf_l2re': surf_l2re if do_val else None,
        'vol_l2re': vol_l2re if do_val else None,
        'epoch_time_s': round(epoch_time, 2),
        'elapsed_s': round(time.time() - start_time, 2),
    }
    with open(os.path.join(path, 'epoch_log.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Latest checkpoint for resume
    if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.nb_epochs - 1:
        save_ckpt('latest', epoch, train_press, train_velo,
                  val_press if do_val else 1e5, val_velo if do_val else 1e5)

    # Best checkpoint by val_loss
    if do_val and val_loss < best_val_loss:
        best_val_loss = val_loss
        save_ckpt('best', epoch, train_press, train_velo, val_press, val_velo)
        logging.info(f'[BEST] New best at epoch {epoch}  val={val_loss:.6f}  '
                     f'surf_l2re={surf_l2re:.4f}  vol_l2re={vol_l2re:.4f}')

    sys.stdout.flush()

time_elapsed = time.time() - start_time
logging.info(f'[DONE] Time: {time_elapsed:.1f}s  best_val={best_val_loss:.6f}')

summary = {
    'nb_parameters': nb_params,
    'time_elapsed_s': round(time_elapsed, 2),
    'hparams': hparams,
    'best_val_loss': best_val_loss,
}
with open(os.path.join(path, f'summary_{args.nb_epochs}.json'), 'w') as f:
    json.dump(summary, f, indent=4)
