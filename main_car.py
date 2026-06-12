import sys
import os
import random
import logging
import argparse
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Transolver', 'Car-Design-ShapeNetCar'))

from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset
import train_airplane as train
from models.Transolver_plus import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/PDE_data/mlcfd_data/training_data')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
args = parser.parse_args()

# dist_nn.all_reduce in Transolver_plus.py requires an initialized process group
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '12355')
backend = 'nccl' if torch.cuda.is_available() else 'gloo'
pass  # distributed not supported on this HPC build

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
if use_cuda:
    torch.cuda.set_device(args.gpu)

# Load and normalize Car-Design data (coef_norm applied inside load_train_val_fold)
train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
train_ds = GraphDataset(train_data)
val_ds = GraphDataset(val_data)


class CarDesignLoader:
    """Converts GraphDataset items to the (x, y, pos, condition, edge) format Transolver_plus expects."""
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
            x = data.x.unsqueeze(0).float()       # (1, N, 7)
            y = data.y.unsqueeze(0).float()        # (1, N, 4)
            pos = data.pos.unsqueeze(0).float()    # (1, N, 3)
            condition = torch.zeros(1, 3)          # (1, 3) — no flow condition for car-design
            yield x, y, pos, condition, None


train_loader = CarDesignLoader(train_ds, shuffle=True)
val_loader = CarDesignLoader(val_ds, shuffle=False)

model = Model(
    n_hidden=256, n_layers=4, space_dim=7,
    fun_dim=0, n_head=8, mlp_ratio=2,
    out_dim=4,      # velo(3) + pressure(1)
    slice_num=32,
    unified_pos=0,
    dropout=0.1,
)

path = f'metrics/car_design/{args.fold_id}/{args.nb_epochs}_{args.weight}'
os.makedirs(path, exist_ok=True)

logging.basicConfig(filename=os.path.join(path, 'train.log'), level=logging.INFO,
                    filemode='w', format='%(asctime)s - %(message)s')
logging.info(args)
logging.info(f'Parameters: {sum(p.numel() for p in model.parameters())}')

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

# pos_norm=0, out_norm=0 because load_train_val_fold already normalized the data
model = train.main(
    device, train_loader, val_loader, model, hparams, path,
    val_iter=args.val_iter, reg=args.weight,
    pos_norm=0, out_norm=0, norm_norm=0,
)
