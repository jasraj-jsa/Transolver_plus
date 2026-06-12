# Running Transolver+ on Kaggle (Free Tier)

This guide covers how to run training on a free Kaggle notebook, handle session limits,
resume after interruptions, and retrieve your logs and checkpoints.

---

## 1. Prerequisites

### GitHub repo
Push this repo to GitHub (public or private).

```bash
git add -A
git commit -m "Add Kaggle checkpointing and logging"
git push origin main
```

### Kaggle dataset with your training data
Upload the `mlcfd_data` folder (or `car_preprocessed`) as a **Kaggle Dataset**:
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Upload your data zip, give it a name like `mlcfd-car-data`
3. Note the dataset slug: `<your-username>/mlcfd-car-data`

---

## 2. Create the Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Settings → **Accelerator: GPU T4 x2** (free tier)
3. Settings → **Internet: On** (needed to clone from GitHub)
4. Add your data dataset: **Add Data** → search `mlcfd-car-data`

---

## 3. Notebook Cell Setup

Paste these cells in order into your notebook.

### Cell 1 — Install dependencies

```python
!pip install -q einops torch_geometric vtk
# torch-geometric extras (match your torch/cuda version — T4 on Kaggle is typically torch 2.x + CUDA 11.8)
!pip install -q torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

> **Note**: Check the exact torch version first with `import torch; print(torch.__version__)`.
> Replace `torch-2.1.0+cu118` accordingly.

### Cell 2 — Clone and set up the repo

```python
import os

REPO_URL = "https://github.com/<your-username>/Transolver_plus.git"
BRANCH = "main"

if not os.path.exists("/kaggle/working/Transolver_plus"):
    !git clone --branch {BRANCH} {REPO_URL} /kaggle/working/Transolver_plus
else:
    # Already cloned in a resumed session — just pull latest
    !git -C /kaggle/working/Transolver_plus pull origin {BRANCH}

# Also clone the parent Transolver repo (needed for Car-Design dataset loader)
TRANSOLVER_URL = "https://github.com/<your-username>/Transolver.git"
if not os.path.exists("/kaggle/working/Transolver"):
    !git clone {TRANSOLVER_URL} /kaggle/working/Transolver

%cd /kaggle/working/Transolver_plus
```

### Cell 3 — Symlink data

```python
# Kaggle mounts datasets at /kaggle/input/<dataset-slug>/
# Adjust the source path to match your uploaded dataset layout.
DATA_INPUT = "/kaggle/input/mlcfd-car-data/training_data"
PREP_INPUT = "/kaggle/input/mlcfd-car-data/car_preprocessed"

# Output (checkpoints, logs) goes to /kaggle/working/ which persists across Save actions
OUTPUT_DIR = "/kaggle/working/transolver_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Data path:", DATA_INPUT)
print("Output path:", OUTPUT_DIR)
```

### Cell 4 — Launch training (first run)

```python
!python main_car.py \
    --data_dir {DATA_INPUT} \
    --save_dir {PREP_INPUT} \
    --output_dir {OUTPUT_DIR} \
    --fold_id 0 \
    --gpu 0 \
    --nb_epochs 200 \
    --val_iter 10 \
    --checkpoint_every 10 \
    --preprocessed 1 \
    --resume 0
```

### Cell 4b — **Resume** after a session limit / crash

Change only `--resume 1`. Everything else stays the same.

```python
!python main_car.py \
    --data_dir {DATA_INPUT} \
    --save_dir {PREP_INPUT} \
    --output_dir {OUTPUT_DIR} \
    --fold_id 0 \
    --gpu 0 \
    --nb_epochs 200 \
    --val_iter 10 \
    --checkpoint_every 10 \
    --preprocessed 1 \
    --resume 1        # <-- only change
```

---

## 4. Understanding the Output Files

All output lands in `OUTPUT_DIR/0/200_0.5/` (fold=0, epochs=200, weight=0.5).

| File | What it contains |
|---|---|
| `train.log` | Full timestamped log of every epoch — opened in **append** mode so resumed runs add to it |
| `epoch_log.jsonl` | One JSON object per epoch: `epoch`, `train_loss`, `val_loss_mse`, `val_loss_l2re`, `epoch_time_s`, `elapsed_s`. Machine-readable. |
| `checkpoint_latest.pth` | Most recent checkpoint — used for resume. Contains model weights + optimizer + scheduler state + epoch number. |
| `checkpoint_best.pth` | Best checkpoint by `val_loss_l2re` seen so far. |
| `summary_200.json` | Written at the very end: total params, elapsed time, final and best metrics. |

### Reading `epoch_log.jsonl` to see progress

```python
import json, pandas as pd

log_path = f"{OUTPUT_DIR}/0/200_0.5/epoch_log.jsonl"
records = [json.loads(l) for l in open(log_path)]
df = pd.DataFrame(records)
print(df.tail(20))
df[['epoch','train_loss','val_loss_l2re']].dropna().plot(x='epoch', title='Training progress')
```

### Tail the text log

```python
!tail -50 {OUTPUT_DIR}/0/200_0.5/train.log
```

---

## 5. Handling Kaggle's Weekly Session Limit

Free Kaggle accounts get **~30 GPU hours / week**. A full 200-epoch run can take longer.

**Strategy**:
1. Run as many epochs as time allows (`--nb_epochs 200` but it will be killed mid-way).
2. The job saves a checkpoint every 10 epochs (configurable with `--checkpoint_every`).
3. **Save the notebook output** before or after the session ends:
   - Click **Save & Run All** or the floppy disk icon to commit outputs.
   - Kaggle persists `/kaggle/working/` for committed notebook versions.
4. When your weekly limit resets (Monday UTC), open the same notebook, run Cell 4b with `--resume 1`.
5. Training picks up from the last saved checkpoint epoch.

> **Important**: Kaggle only persists `/kaggle/working/` if you **Save** the notebook version.
> Uncommitted sessions may lose `/kaggle/working/` on timeout. Save early, save often.

### Reduce epochs per session (safer approach)

Instead of 200 at once, run in segments:

```python
# Week 1: epochs 0–79
!python main_car.py ... --nb_epochs 80 --resume 0

# Week 2: resume and run to 200
!python main_car.py ... --nb_epochs 200 --resume 1
```

The checkpoint stores the completed epoch number, so `--nb_epochs 200` with `--resume 1`
will skip already-done epochs automatically.

---

## 6. Downloading Results

### Option A — Kaggle output download
After the notebook run is saved, go to the notebook → **Output** tab → download files.

### Option B — From another notebook

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
# download all output files from a specific notebook version
api.kernels_output('<username>/<notebook-slug>', path='/local/download/path')
```

### Option C — Sync to Google Drive (add to notebook)

```python
from google.colab import drive  # won't work on Kaggle directly — use gdown or rclone instead
!pip install -q rclone
# configure rclone with your Google Drive token, then:
!rclone copy {OUTPUT_DIR} gdrive:/TransolverPlusResults --progress
```

---

## 7. Tips for Free Tier

- Use `--val_iter 20` instead of 10 to spend less time on validation and more on training.
- `--checkpoint_every 5` if you're worried about frequent crashes (more disk I/O but safer).
- Monitor GPU memory: `!nvidia-smi` in a cell.
- If the kernel goes OOM, reduce batch size: `--batch_size 1` (already the default).
- The `checkpoint_best.pth` is your safety net — even if the final epoch never runs, you have the best model found so far.

---

## 8. Loading a Checkpoint for Inference

```python
import torch, sys, os
sys.path.insert(0, '/kaggle/working/Transolver_plus')
sys.path.insert(0, '/kaggle/working/Transolver/Car-Design-ShapeNetCar')
from models.Transolver_plus import Model

model = Model(
    n_hidden=256, n_layers=4, space_dim=7,
    fun_dim=0, n_head=8, mlp_ratio=2,
    out_dim=4, slice_num=32, unified_pos=0, dropout=0.1,
)

ckpt = torch.load('/kaggle/working/transolver_output/0/200_0.5/checkpoint_best.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}  val_l2re={ckpt['val_loss_l2re']:.6f}")
```
