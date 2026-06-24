"""
benchmark_car.py — Honest comparison of Transolver vs Transolver+ on ShapeNet Car.

Three experiments in one script:
  A. Transolver (original, standard softmax) vs Transolver+ (Gumbel-softmax)
     — both at 8 layers, same hyperparams, same data. The core paper claim.
  B. Depth ablation: 4 vs 8 layers on Transolver+
     — shows whether Gumbel gains scale with model depth on this task.
  C. Slice count ablation: 32 vs 64 slices on Transolver+ (8 layers)
     — more slices = finer physics partitioning.

Each experiment trains from scratch and saves its result to results/benchmark/.
Run with --experiment A, B, or C (or 'all' to run sequentially).

Usage:
    python benchmark_car.py --experiment A --data_dir ./dataset/mlcfd_data/car_preprocessed --gpu 0
    python benchmark_car.py --experiment B --data_dir ./dataset/mlcfd_data/car_preprocessed --gpu 0
    python benchmark_car.py --experiment C --data_dir ./dataset/mlcfd_data/car_preprocessed --gpu 0
"""

import os, sys, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Transolver', 'Car-Design-ShapeNetCar'))
from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='A', choices=['A', 'B', 'C', 'all'])
parser.add_argument('--data_dir',   default='./dataset/mlcfd_data/car_preprocessed')
parser.add_argument('--gpu',        default=0, type=int)
parser.add_argument('--epochs',     default=200, type=int)
parser.add_argument('--fold_id',    default=0, type=int)
parser.add_argument('--lr',         default=1e-3, type=float)
parser.add_argument('--weight',     default=0.5, type=float)
args = parser.parse_args()

args.save_dir = args.data_dir
args.preprocessed = 1
args.batch_size = 1

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {device}')

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results/benchmark', exist_ok=True)

# ── Data (loaded once, shared across experiments) ──────────────────────────
print('[INFO] Loading data …')
train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=1)
train_ds = GraphDataset(train_data)
val_ds   = GraphDataset(val_data)
print(f'[INFO] Train: {len(train_ds)}  Val: {len(val_ds)}')


class CarLoader:
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
            yield (d.x.unsqueeze(0).float(), d.y.unsqueeze(0).float(),
                   d.pos.unsqueeze(0).float(), torch.zeros(1, 3), d.surf)


# ── Model builders ─────────────────────────────────────────────────────────
def build_transolver_plus(n_layers=8, n_hidden=256, slice_num=32):
    """Transolver+ with Gumbel-softmax slice assignment (the new model)."""
    from models.Transolver_plus import Model
    return Model(space_dim=7, n_layers=n_layers, n_hidden=n_hidden,
                 dropout=0.0, n_head=8, mlp_ratio=2, fun_dim=0,
                 out_dim=4, slice_num=slice_num, unified_pos=0)

def build_transolver(n_layers=8, n_hidden=256, slice_num=32):
    """Original Transolver with standard softmax slice assignment."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    '..', 'Transolver', 'Car-Design-ShapeNetCar'))
    from models.Transolver import Model
    return Model(space_dim=7, n_layers=n_layers, n_hidden=n_hidden,
                 dropout=0.0, n_head=8, mlp_ratio=2, fun_dim=0,
                 out_dim=4, slice_num=slice_num, unified_pos=0)


# ── Training and evaluation ─────────────────────────────────────────────────
criterion = nn.MSELoss(reduction='none')

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    press_losses, velo_losses = [], []
    for x, y, pos, cond, surf in loader:
        x, y, pos, cond, surf = (t.to(device) for t in (x, y, pos, cond, surf))
        optimizer.zero_grad()
        out = model((x, pos, cond)).squeeze(0)
        tgt = y.squeeze(0)
        loss_p = criterion(out[surf,  -1], tgt[surf,  -1]).mean()
        loss_v = criterion(out[:, :3],     tgt[:, :3]).mean()
        (loss_v + args.weight * loss_p).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        press_losses.append(loss_p.item())
        velo_losses.append(loss_v.item())
    return float(np.mean(velo_losses)) + args.weight * float(np.mean(press_losses))

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    surf_l2re, vol_l2re = [], []
    for x, y, pos, cond, surf in loader:
        x, y, pos, cond, surf = (t.to(device) for t in (x, y, pos, cond, surf))
        out = model((x, pos, cond)).squeeze(0)
        tgt = y.squeeze(0)
        surf_l2re.append((torch.norm(out[surf,  -1] - tgt[surf,  -1]) /
                          (torch.norm(tgt[surf,  -1]) + 1e-8)).item())
        vol_l2re.append( (torch.norm(out[~surf, :3]  - tgt[~surf, :3]) /
                          (torch.norm(tgt[~surf, :3])  + 1e-8)).item())
    return float(np.mean(surf_l2re)), float(np.mean(vol_l2re))

def run_experiment(name, model, label):
    """Full train + eval loop for one model config. Returns history dict."""
    print(f'\n{"="*60}')
    print(f'Running: {label}')
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {nb_params:,}')
    print(f'{"="*60}')

    model = model.to(device)
    train_loader = CarLoader(train_ds, shuffle=True)
    val_loader   = CarLoader(val_ds,   shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=int((len(train_ds) + 1) * args.epochs),
        final_div_factor=1000.,
    )

    history = []
    best_surf, best_vol, best_combined = 1e5, 1e5, 1e5

    for ep in tqdm(range(args.epochs), desc=label):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)

        if ep % 10 == 0 or ep == args.epochs - 1:
            surf_l2re, vol_l2re = evaluate(model, val_loader)
            combined = surf_l2re + vol_l2re
            history.append({'epoch': ep, 'train_loss': train_loss,
                            'surf_l2re': surf_l2re, 'vol_l2re': vol_l2re})
            tqdm.write(f'  ep={ep:>3}  train={train_loss:.5f}  '
                       f'surf={surf_l2re:.4f}  vol={vol_l2re:.4f}')
            if combined < best_combined:
                best_combined = combined
                best_surf, best_vol = surf_l2re, vol_l2re
                torch.save(model.state_dict(), f'./checkpoints/{name}.pt')
        else:
            history.append({'epoch': ep, 'train_loss': train_loss,
                            'surf_l2re': None, 'vol_l2re': None})

    result = {
        'name': name, 'label': label, 'nb_params': nb_params,
        'n_layers': getattr(model, 'n_layers', '?'),
        'best_surf_l2re': best_surf, 'best_vol_l2re': best_vol,
        'epochs': args.epochs, 'history': history,
        'paper_baseline': {'model': 'Transolver', 'surf': 0.0745, 'vol': 0.0207},
    }
    with open(f'./results/benchmark/{name}.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n  Best Surf L2RE : {best_surf:.4f}  (Transolver baseline: 0.0745)')
    print(f'  Best Vol  L2RE : {best_vol:.4f}  (Transolver baseline: 0.0207)')
    return result


# ── Experiment definitions ─────────────────────────────────────────────────
EXPERIMENTS = {
    'A': [
        ('transolver_L8',      build_transolver(n_layers=8),       'Transolver  (8L, standard softmax)'),
        ('transolver_plus_L8', build_transolver_plus(n_layers=8),  'Transolver+ (8L, Gumbel-softmax)'),
    ],
    'B': [
        ('transolver_plus_L4', build_transolver_plus(n_layers=4),  'Transolver+ (4L, Gumbel-softmax)'),
        ('transolver_plus_L8', build_transolver_plus(n_layers=8),  'Transolver+ (8L, Gumbel-softmax)'),
    ],
    'C': [
        ('transolver_plus_L8_S32', build_transolver_plus(n_layers=8, slice_num=32), 'Transolver+ (8L, 32 slices)'),
        ('transolver_plus_L8_S64', build_transolver_plus(n_layers=8, slice_num=64), 'Transolver+ (8L, 64 slices)'),
    ],
}

to_run = []
if args.experiment == 'all':
    for exp_runs in EXPERIMENTS.values():
        to_run += exp_runs
    # Deduplicate by name (B and A share transolver_plus_L8)
    seen = set()
    to_run = [(n, m, l) for n, m, l in to_run if not (n in seen or seen.add(n))]
else:
    to_run = EXPERIMENTS[args.experiment]

results = {}
for name, model, label in to_run:
    results[name] = run_experiment(name, model, label)


# ── Summary plot and table ─────────────────────────────────────────────────
def make_summary_plot(results):
    val_res = {k: v for k, v in results.items() if v['best_surf_l2re'] < 1e4}
    if len(val_res) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    for ax, metric, baseline, title in [
        (axes[0], 'surf_l2re', 0.0745, 'Surf L2RE (surface pressure)'),
        (axes[1], 'vol_l2re',  0.0207, 'Volume L2RE (surrounding velocity)'),
    ]:
        for i, (name, res) in enumerate(val_res.items()):
            epochs = [r['epoch'] for r in res['history'] if r.get(metric) is not None]
            vals   = [r[metric]  for r in res['history'] if r.get(metric) is not None]
            ax.plot(epochs, vals, label=res['label'], color=colors[i % len(colors)],
                    marker='o', markersize=3)
        ax.axhline(baseline, color='black', linestyle='--', linewidth=1.5,
                   label=f'Transolver baseline ({baseline})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative L2')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Transolver+ Benchmark — ShapeNet Car', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('./results/benchmark/comparison.png', dpi=150)
    plt.close(fig)
    print('\nSaved ./results/benchmark/comparison.png')

def print_table(results):
    print(f'\n{"="*70}')
    print(f'{"BENCHMARK RESULTS — ShapeNet Car":^70}')
    print(f'{"="*70}')
    print(f'  {"Model":<38} {"Surf L2RE":>10} {"Vol L2RE":>10} {"Params":>10}')
    print(f'  {"-"*68}')
    for name, res in results.items():
        if res['best_surf_l2re'] < 1e4:
            print(f'  {res["label"]:<38} {res["best_surf_l2re"]:>10.4f} '
                  f'{res["best_vol_l2re"]:>10.4f} {res["nb_params"]:>10,}')
    print(f'  {"-"*68}')
    print(f'  {"Transolver (paper, Table 3)":<38} {"0.0745":>10} {"0.0207":>10} {"~1.4M":>10}')
    print(f'{"="*70}')

make_summary_plot(results)
print_table(results)

# Merge with any prior benchmark results for the full table
all_results = {}
import glob
for f in glob.glob('./results/benchmark/*.json'):
    try:
        d = json.load(open(f))
        all_results[d['name']] = d
    except Exception:
        pass
if len(all_results) > len(results):
    print('\n--- Including all prior benchmark runs: ---')
    print_table(all_results)
    make_summary_plot(all_results)
