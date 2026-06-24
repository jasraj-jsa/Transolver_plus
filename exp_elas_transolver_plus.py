"""
exp_elas_transolver_plus.py — Elasticity benchmark for Transolver+.

Mirrors Transolver/PDE-Solving-StandardBenchmark/exp_elas.py so results
are directly comparable. Uses Physics_Attention_1D_Eidetic (Gumbel-softmax
slicing) instead of the original softmax-based Physics_Attention.

Attention saving:
  During eval the per-layer slice-to-slice attention matrices (B H G G) are
  collected from each Transolver_plus_block.Attn.last_attn and stacked into
  a single array of shape (n_layers, H, G, G), then saved to:
    ./results/<save_name>/elasticity_attn.npy

Usage:
    # Train
    python exp_elas_transolver_plus.py \
        --data_path /kaggle/input/fno \
        --gpu 0 --n-layers 8 --n-hidden 128 --slice_num 64

    # Eval only (loads checkpoint, saves attn)
    python exp_elas_transolver_plus.py \
        --data_path /kaggle/input/fno \
        --gpu 0 --eval 1 --save_name elas_Transolver_plus
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reuse normalizer and loss from the original Transolver benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Transolver',
                                'PDE-Solving-StandardBenchmark'))
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer

from models.Transolver_plus import Model

# ── Args (mirrors exp_elas.py) ─────────────────────────────────────────────
parser = argparse.ArgumentParser('Transolver+ Elasticity Benchmark')
parser.add_argument('--lr',            type=float, default=1e-3)
parser.add_argument('--epochs',        type=int,   default=500)
parser.add_argument('--weight_decay',  type=float, default=1e-5)
parser.add_argument('--n-hidden',      type=int,   default=128)
parser.add_argument('--n-layers',      type=int,   default=8)
parser.add_argument('--n-heads',       type=int,   default=8)
parser.add_argument('--batch-size',    type=int,   default=8)
parser.add_argument('--mlp_ratio',     type=int,   default=1)
parser.add_argument('--dropout',       type=float, default=0.0)
parser.add_argument('--ntrain',        type=int,   default=1000)
parser.add_argument('--slice_num',     type=int,   default=64)
parser.add_argument('--unified_pos',   type=int,   default=0)
parser.add_argument('--ref',           type=int,   default=8)
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--gpu',           type=str,   default='0')
parser.add_argument('--eval',          type=int,   default=0)
parser.add_argument('--save_name',     type=str,   default='elas_Transolver_plus')
parser.add_argument('--data_path',     type=str,   default='/data/fno')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(args)

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results/' + args.save_name, exist_ok=True)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Trainable Params: {total:,}')
    return total


# ── Data ───────────────────────────────────────────────────────────────────
PATH_Sigma = os.path.join(args.data_path, 'elasticity/Meshes/Random_UnitCell_sigma_10.npy')
PATH_XY    = os.path.join(args.data_path, 'elasticity/Meshes/Random_UnitCell_XY_10.npy')

input_s  = torch.tensor(np.load(PATH_Sigma), dtype=torch.float).permute(1, 0)
input_xy = torch.tensor(np.load(PATH_XY),    dtype=torch.float).permute(2, 0, 1)

ntrain, ntest = args.ntrain, 200
train_s,  test_s  = input_s[:ntrain],  input_s[-ntest:]
train_xy, test_xy = input_xy[:ntrain], input_xy[-ntest:]

print(input_s.shape, input_xy.shape)

y_normalizer = UnitTransformer(train_s)
train_s = y_normalizer.encode(train_s)
y_normalizer.cuda()

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_xy, train_xy, train_s),
    batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_xy, test_xy, test_s),
    batch_size=1, shuffle=False)

print('Dataloading is over.')

# ── Model ──────────────────────────────────────────────────────────────────
# Elasticity: irregular mesh, 2D coords as input, scalar stress output.
# Transolver+ uses space_dim to set input feature size via preprocess MLP.
model = Model(
    space_dim=2,
    n_layers=args.n_layers,
    n_hidden=args.n_hidden,
    dropout=args.dropout,
    n_head=args.n_heads,
    mlp_ratio=args.mlp_ratio,
    fun_dim=0,
    out_dim=1,
    slice_num=args.slice_num,
    ref=args.ref,
    unified_pos=args.unified_pos,
).to(device)

print(args)
print(model)
count_parameters(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=args.epochs)
myloss = TestLoss(size_average=False)


def _collect_attn(model):
    """Return stacked last_attn from all blocks: (L, B, H, G, G)."""
    attns = []
    for block in model.blocks:
        a = block.Attn.last_attn
        if a is not None:
            attns.append(a.detach().cpu())
    if not attns:
        return None
    return torch.stack(attns, dim=0).numpy()  # (L, B, H, G, G)


# ── Eval-only mode ─────────────────────────────────────────────────────────
if args.eval:
    ckpt = os.path.join('./checkpoints', args.save_name + '.pt')
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()

    result_dir = './results/' + args.save_name + '/'
    rel_err = 0.0
    attn_accum = []

    with torch.no_grad():
        for idx, (pos, fx, y) in enumerate(test_loader):
            pos, fx, y = pos.to(device), fx.to(device), y.to(device)

            # Transolver+ Model.forward expects (x, pos, condition)
            # For elasticity: x = (pos, fx=pos), condition = None
            out = model((pos, pos, None)).squeeze(-1)
            out = y_normalizer.decode(out)

            # collect attn from this forward pass (shape L B H G G, B=1 here)
            a = _collect_attn(model)
            if a is not None:
                attn_accum.append(a[:, 0])  # drop batch dim → (L, H, G, G)

            tl = myloss(out, y).item()
            rel_err += tl

            if idx < 10:
                plt.axis('off')
                plt.scatter(fx[0, :, 0].cpu(), fx[0, :, 1].cpu(),
                            c=y[0].cpu(), cmap='coolwarm')
                plt.colorbar()
                plt.clim(0, 1000)
                plt.savefig(os.path.join(result_dir, f'gt_{idx+1}.pdf'),
                            bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.axis('off')
                plt.scatter(fx[0, :, 0].cpu(), fx[0, :, 1].cpu(),
                            c=out[0].detach().cpu(), cmap='coolwarm')
                plt.colorbar()
                plt.clim(0, 1000)
                plt.savefig(os.path.join(result_dir, f'pred_{idx+1}.pdf'),
                            bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.axis('off')
                plt.scatter(fx[0, :, 0].cpu(), fx[0, :, 1].cpu(),
                            c=(y[0] - out[0]).detach().cpu(), cmap='coolwarm')
                plt.colorbar()
                plt.clim(-8, 8)
                plt.savefig(os.path.join(result_dir, f'error_{idx+1}.pdf'),
                            bbox_inches='tight', pad_inches=0)
                plt.close()

    rel_err /= ntest
    print(f'rel_err : {rel_err}')

    # Save attention — stack across test samples: (n_samples, L, H, G, G)
    if attn_accum:
        attn_arr = np.stack(attn_accum, axis=0)
        # Also save mean across samples for easy plotting, matching
        # the Transolver Kaggle snippet: mean over batch → (L, H, G, G)
        attn_mean = attn_arr.mean(axis=0)  # (L, H, G, G)
        np.save(os.path.join(result_dir, 'elasticity_attn.npy'), attn_mean)
        print(f'Attention saved to {result_dir}elasticity_attn.npy  '
              f'shape={attn_mean.shape}')

    sys.exit(0)

# ── Training ───────────────────────────────────────────────────────────────
for ep in range(args.epochs):
    model.train()
    train_loss = 0.0

    for pos, fx, y in train_loader:
        pos, fx, y = pos.to(device), fx.to(device), y.to(device)
        optimizer.zero_grad()
        out = model((pos, pos, None)).squeeze(-1)
        out = y_normalizer.decode(out)
        y   = y_normalizer.decode(y)
        loss = myloss(out, y)
        loss.backward()
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_loss /= ntrain
    print(f'Epoch {ep}  Train loss : {train_loss:.5f}')

    model.eval()
    rel_err = 0.0
    with torch.no_grad():
        for pos, fx, y in test_loader:
            pos, fx, y = pos.to(device), fx.to(device), y.to(device)
            out = model((pos, pos, None)).squeeze(-1)
            out = y_normalizer.decode(out)
            rel_err += myloss(out, y).item()
    rel_err /= ntest
    print(f'rel_err : {rel_err:.5f}')

    if ep % 100 == 0:
        torch.save(model.state_dict(),
                   os.path.join('./checkpoints', args.save_name + '.pt'))
        print('save model')

torch.save(model.state_dict(),
           os.path.join('./checkpoints', args.save_name + '.pt'))
print('save model')
