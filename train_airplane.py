import numpy as np
import time, json, os, sys
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def flush_logs():
    """Flush all logging handlers and stdout/stderr — important for Kaggle."""
    for handler in logging.root.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss_mse, val_loss_l2re, path, tag='latest'):
    """Save a resumable checkpoint with full optimizer/scheduler state."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss_mse': val_loss_mse,
        'val_loss_l2re': val_loss_l2re,
    }
    ckpt_path = os.path.join(path, f'checkpoint_{tag}.pth')
    torch.save(ckpt, ckpt_path)
    logging.info(f'[CHECKPOINT] Saved {tag} checkpoint at epoch {epoch} → {ckpt_path}')
    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, path, tag='latest'):
    """Load checkpoint and return the last completed epoch (-1 if none found)."""
    ckpt_path = os.path.join(path, f'checkpoint_{tag}.pth')
    if not os.path.exists(ckpt_path):
        logging.info(f'[CHECKPOINT] No checkpoint found at {ckpt_path}, starting from scratch.')
        return -1, 1e5, 1e5, 1e5
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt['epoch']
    train_loss = ckpt.get('train_loss', 1e5)
    val_loss_mse = ckpt.get('val_loss_mse', 1e5)
    val_loss_l2re = ckpt.get('val_loss_l2re', 1e5)
    logging.info(f'[CHECKPOINT] Resumed from epoch {epoch}  train_loss={train_loss:.6f}  val_mse={val_loss_mse:.6f}  val_l2re={val_loss_l2re:.6f}')
    print(f'[CHECKPOINT] Resumed from epoch {epoch}  train_loss={train_loss:.6f}  val_mse={val_loss_mse:.6f}  val_l2re={val_loss_l2re:.6f}')
    return epoch, train_loss, val_loss_mse, val_loss_l2re


def append_epoch_log(path, record):
    """Append a single epoch record to the JSONL epoch log (one JSON object per line)."""
    log_path = os.path.join(path, 'epoch_log.jsonl')
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def train(device, model, train_loader, optimizer, scheduler, reg=1, pos_norm=0, norm_norm=0, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses_mse = []
    for x, y, pos, geom, edge in train_loader:
        x = x.to(device)
        pos = pos.to(device)
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        y = y.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
        out = model((x, pos, geom))

        if out_norm:
            y = (y - out_mean) / (out_std + 1e-6)

        loss_press = criterion_func(out, y).mean()
        total_loss = loss_press
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses_mse.append(loss_press.item())

    return np.mean(losses_mse)


@torch.no_grad()
def test(device, model, test_loader, pos_norm=0, norm_norm=1, out_norm=1, pos_mean=None, pos_std=None, norm_mean=None, norm_std=None, out_mean=None, out_std=None, full=False):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_mse = []
    losses_l2re = []

    for x, y, pos, geom, edge in test_loader:
        x = x.to(device)
        pos = pos.to(device)
        if pos_norm:
            pos = (pos - pos_mean) / pos_std
            x[:, :, :3] = pos
        y = y.to(device)
        if geom is not None:
            geom = geom.to(device)
        out = model((x, pos, geom))

        if out_norm:
            y_norm = (y - out_mean) / (out_std + 1e-6)
            loss_mse = criterion_func(out, y_norm).mean()
            out = out * out_std + out_mean
            loss_l2re = torch.norm(out[:, :, -1] - y[:, :, -1]) / torch.norm(y[:, :, -1])
        else:
            loss_mse = criterion_func(out[:, :, -1], y[:, :, -1]).mean()
            loss_l2re = torch.norm(out[:, :, -1] - y[:, :, -1]) / torch.norm(y[:, :, -1])
        losses_mse.append(loss_mse.item())
        losses_l2re.append(loss_l2re.item())

    return np.mean(losses_mse), np.mean(losses_l2re)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_loader, val_loader, Net, hparams, path, reg=1, val_iter=1,
         pos_norm=0, out_norm=1, norm_norm=0,
         pos_mean=None, pos_std=None, out_mean=None, out_std=None,
         norm_mean=None, norm_std=None, full=False,
         resume=False, checkpoint_every=10):
    """
    Main training loop.

    resume          – if True, attempt to load 'checkpoint_latest.pth' before training.
    checkpoint_every – save a checkpoint every N epochs (also always saves 'best' by val_loss_l2re).
    """
    model = Net.to(device)
    nb_epochs = hparams['nb_epochs']

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=int((len(train_loader) // hparams['batch_size'] + 1) * nb_epochs),
        final_div_factor=1000.,
    )

    start_epoch = 0
    train_loss, val_loss_mse, val_loss_l2re = 1e5, 1e5, 1e5
    best_val_l2re = 1e5

    if resume:
        last_epoch, train_loss, val_loss_mse, val_loss_l2re = load_checkpoint(
            model, optimizer, lr_scheduler, path, tag='latest'
        )
        if last_epoch >= 0:
            start_epoch = last_epoch + 1
            best_val_l2re = val_loss_l2re

    params_model = get_nb_trainable_params(model)
    print(f'[INFO] Model parameters: {int(params_model):,}')
    print(f'[INFO] Training epochs: {start_epoch} → {nb_epochs - 1}')
    print(f'[INFO] Checkpoint dir: {path}')
    logging.info(f'[INFO] Model parameters: {int(params_model):,}')
    logging.info(f'[INFO] Training from epoch {start_epoch} to {nb_epochs - 1}')
    logging.info(f'[INFO] checkpoint_every={checkpoint_every}  val_iter={val_iter}')
    flush_logs()

    start = time.time()
    pbar_train = tqdm(range(start_epoch, nb_epochs), position=0, initial=start_epoch, total=nb_epochs)

    for epoch in pbar_train:
        epoch_start = time.time()
        loss_mse = train(
            device, model, train_loader, optimizer, lr_scheduler,
            reg=reg, pos_norm=pos_norm, out_norm=out_norm, norm_norm=norm_norm,
            pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std,
            norm_mean=norm_mean, norm_std=norm_std, full=full
        )
        train_loss = loss_mse
        epoch_time = time.time() - epoch_start

        do_val = val_iter is not None and (epoch == nb_epochs - 1 or epoch % val_iter == 0)
        if do_val:
            val_loss_mse, val_loss_l2re = test(
                device, model, val_loader,
                pos_norm=pos_norm, out_norm=out_norm, norm_norm=norm_norm,
                pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std,
                norm_mean=norm_mean, norm_std=norm_std, full=full
            )

            pbar_train.set_postfix(train_loss=f'{train_loss:.5f}', val_mse=f'{val_loss_mse:.5f}', val_l2re=f'{val_loss_l2re:.5f}')
            msg = (f'Epoch {epoch}/{nb_epochs-1}  train_loss={train_loss:.6f}  '
                   f'val_mse={val_loss_mse:.6f}  val_l2re={val_loss_l2re:.6f}  '
                   f'epoch_time={epoch_time:.1f}s')
        else:
            pbar_train.set_postfix(train_loss=f'{train_loss:.5f}')
            msg = f'Epoch {epoch}/{nb_epochs-1}  train_loss={train_loss:.6f}  epoch_time={epoch_time:.1f}s'

        print(msg)
        logging.info(msg)

        # Per-epoch JSONL log — always written, so you have a full trace even if job dies
        append_epoch_log(path, {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss_mse': float(val_loss_mse) if do_val else None,
            'val_loss_l2re': float(val_loss_l2re) if do_val else None,
            'epoch_time_s': round(epoch_time, 2),
            'elapsed_s': round(time.time() - start, 2),
        })

        # Rolling latest checkpoint (always keep this one for resume)
        if (epoch + 1) % checkpoint_every == 0 or epoch == nb_epochs - 1:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, train_loss,
                            val_loss_mse, val_loss_l2re, path, tag='latest')

        # Best-model checkpoint by val L2RE
        if do_val and val_loss_l2re < best_val_l2re:
            best_val_l2re = val_loss_l2re
            save_checkpoint(model, optimizer, lr_scheduler, epoch, train_loss,
                            val_loss_mse, val_loss_l2re, path, tag='best')
            print(f'[CHECKPOINT] New best model at epoch {epoch}  val_l2re={val_loss_l2re:.6f}')
            logging.info(f'[CHECKPOINT] New best model at epoch {epoch}  val_l2re={val_loss_l2re:.6f}')

        flush_logs()

    end = time.time()
    time_elapsed = end - start

    print(f'[DONE] Training complete.  Total time: {time_elapsed:.1f}s')
    print(f'[DONE] Best val_l2re: {best_val_l2re:.6f}')
    logging.info(f'[DONE] Training complete.  Total time: {time_elapsed:.1f}s')
    logging.info(f'[DONE] Best val_l2re: {best_val_l2re:.6f}')

    # Final summary JSON
    summary = {
        'nb_parameters': int(params_model),
        'time_elapsed_s': round(time_elapsed, 2),
        'hparams': hparams,
        'final_train_loss': float(train_loss),
        'final_val_loss_mse': float(val_loss_mse),
        'final_val_loss_l2re': float(val_loss_l2re),
        'best_val_l2re': float(best_val_l2re),
    }
    summary_path = os.path.join(path, f'summary_{nb_epochs}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)
    print(f'[DONE] Summary written to {summary_path}')
    logging.info(f'[DONE] Summary written to {summary_path}')
    flush_logs()

    return model
