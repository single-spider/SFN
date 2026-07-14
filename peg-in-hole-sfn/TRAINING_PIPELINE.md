# Robust Perception Training Pipeline

This runbook is for long terminal runs.  The scripts now support progress bars,
resume checkpoints, validation metrics, early stopping, chunked datasets, edge
cases, and small serial hyper-parameter searches.

Current research default: use shape-disjoint splits with about 75% of shapes in
training and the rest held out for validation/test.  Limited training runs must
use `--limit-strategy stratified`; prefix limits are only for debugging because
the chunked datasets are written shape-by-shape and a prefix can silently select
a single shape.

After changing the split or dataset identity, start perception runs from scratch
unless the checkpoint was trained with the same model code, split, and dataset.

Run from:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
```

## 1. Build larger datasets

Training split with edge/boundary poses and chunked NPZ files.  The default
`train_seen` split is now 12 of the 16 synthetic shapes (75%).  Regenerate this
dataset after changing split constants; older `train_seen_40k_edge_fast`
datasets contain only the previous four training shapes.

```powershell
..\.venv\Scripts\python.exe scripts\collect_dataset.py `
  --split train_seen `
  --samples-per-shape 10000 `
  --chunk-size 250 `
  --include-edge-cases `
  --progress-every 100 `
  --out data\train_seen_120k_edge `
  --seed 100
```

If collection appears slow at chunk-writing time, add `--no-compress`.  This is
much faster but uses more disk:

```powershell
..\.venv\Scripts\python.exe scripts\collect_dataset.py `
  --split train_seen `
  --samples-per-shape 10000 `
  --chunk-size 250 `
  --include-edge-cases `
  --progress-every 100 `
  --no-compress `
  --out data\train_seen_120k_edge_fast `
  --seed 100
```

Validation split:

```powershell
..\.venv\Scripts\python.exe scripts\collect_dataset.py `
  --split validation_unseen `
  --samples-per-shape 2000 `
  --chunk-size 250 `
  --include-edge-cases `
  --progress-every 100 `
  --no-compress `
  --out data\val_unseen_4k_edge_fast `
  --seed 200
```

Orientation-friendly validation split.  Use this for position/orientation
training/evaluation; it was generated after fixing a renderer edge case where
one validation peg was perfectly square and therefore had no visible yaw signal:

```powershell
..\.venv\Scripts\python.exe scripts\collect_dataset.py `
  --split validation_unseen `
  --samples-per-shape 2000 `
  --chunk-size 250 `
  --include-edge-cases `
  --progress-every 500 `
  --no-compress `
  --out data\val_unseen_4k_edge_orientable `
  --seed 200
```

Test split:

```powershell
..\.venv\Scripts\python.exe scripts\collect_dataset.py `
  --split test_unseen `
  --samples-per-shape 2000 `
  --chunk-size 250 `
  --include-edge-cases `
  --progress-every 100 `
  --out data\test_unseen_4k_edge `
  --seed 300
```

Validate each dataset:

```powershell
..\.venv\Scripts\python.exe scripts\validate_dataset.py data\train_seen_120k_edge
..\.venv\Scripts\python.exe scripts\validate_dataset.py data\train_seen_120k_edge_fast
..\.venv\Scripts\python.exe scripts\validate_dataset.py data\val_unseen_4k_edge_fast
..\.venv\Scripts\python.exe scripts\validate_dataset.py data\test_unseen_4k_edge
```

## 2. Train segmentation

```powershell
..\.venv\Scripts\python.exe scripts\train_segmentation.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_fast `
  --out models\segmentation.pt `
  --epochs 50 `
  --batch-size 16 `
  --lr 0.0003 `
  --loss focal `
  --class-weight median `
  --base-channels 32 `
  --patience 8 `
  --seed 101
```

Resume if interrupted:

```powershell
..\.venv\Scripts\python.exe scripts\train_segmentation.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_fast `
  --out models\segmentation.pt `
  --epochs 50 `
  --batch-size 16 `
  --lr 0.0003 `
  --loss focal `
  --class-weight median `
  --base-channels 32 `
  --resume models\segmentation.last.pt `
  --patience 8 `
  --seed 101
```

Outputs:

```text
models\segmentation.pt              best checkpoint
models\segmentation.last.pt         latest checkpoint for resume
models\segmentation.metrics.jsonl   one JSON metrics row per epoch
models\segmentation.summary.json    final summary
```

Resume note: `--epochs` is the total target epoch count, not an additional
epoch count.  For example, a checkpoint saved at epoch 12 resumes to epoch 50
with `--epochs 50`.

## 3. Train position

```powershell
..\.venv\Scripts\python.exe scripts\train_position.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_orientable `
  --out models\position.pt `
  --epochs 3 `
  --batch-size 128 `
  --lr 0.0003 `
  --base-channels 32 `
  --device cuda `
  --amp `
  --limit 4096 `
  --limit-strategy stratified `
  --patience 2 `
  --seed 102
```

## 4. Train orientation

```powershell
..\.venv\Scripts\python.exe scripts\train_orientation.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_orientable `
  --out models\orientation.pt `
  --epochs 3 `
  --batch-size 128 `
  --lr 0.0003 `
  --base-channels 32 `
  --device cuda `
  --amp `
  --limit 4096 `
  --limit-strategy stratified `
  --patience 2 `
  --seed 103
```

## 5. Hyper-parameter search

Example segmentation search:

```powershell
..\.venv\Scripts\python.exe scripts\train_segmentation.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_fast `
  --out artifacts\seg_search\seg.pt `
  --epochs 12 `
  --batch-size 16 `
  --search-grid "lr=0.001,0.0003;loss=weighted_ce,focal;base_channels=16,32" `
  --max-trials 8 `
  --patience 4 `
  --seed 500
```

Search outputs:

```text
artifacts\seg_search\seg\search_results.csv
artifacts\seg_search\seg\search_summary.json
```

## 6. Evaluate trained perception

```powershell
..\.venv\Scripts\python.exe scripts\evaluate_perception.py `
  --dataset data\test_unseen_4k_edge `
  --segmentation models\segmentation.pt `
  --position models\position.pt `
  --orientation models\orientation.pt `
  --per-shape `
  --out artifacts\perception_test\metrics.json
```

## 7. Scheduler-friendly long-run workflow

For multi-hour runs, prefer your scheduler over having an interactive agent
poll the terminal every few minutes.  The training scripts already write durable
per-epoch metrics:

```text
models\position.metrics.jsonl
models\orientation.metrics.jsonl
models\*.last.pt
models\*.summary.json
```

Recommended cadence:

```text
check every 30-60 minutes
do not judge before the first full epoch finishes
stop/reconfigure only when metrics are clearly bad after at least one completed epoch
```

### 10-epoch position batch

Use `--no-progress` when redirecting logs from a scheduler.  Use `--val-limit
1024` for periodic checks so validation does not dominate runtime; run full
evaluation later with `scripts\evaluate_perception.py`.

```powershell
New-Item -ItemType Directory -Force -Path artifacts\scheduled_runs | Out-Null

..\.venv\Scripts\python.exe scripts\train_position.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_orientable `
  --out models\position.pt `
  --epochs 10 `
  --batch-size 128 `
  --lr 0.0003 `
  --base-channels 32 `
  --device cuda `
  --amp `
  --limit 4096 `
  --val-limit 1024 `
  --limit-strategy stratified `
  --patience 3 `
  --no-progress `
  --seed 102 *> artifacts\scheduled_runs\position_10epoch.log
```

### 10-epoch orientation batch

```powershell
New-Item -ItemType Directory -Force -Path artifacts\scheduled_runs | Out-Null

..\.venv\Scripts\python.exe scripts\train_orientation.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_orientable `
  --out models\orientation.pt `
  --epochs 10 `
  --batch-size 128 `
  --lr 0.0003 `
  --base-channels 32 `
  --device cuda `
  --amp `
  --limit 4096 `
  --val-limit 1024 `
  --limit-strategy stratified `
  --patience 3 `
  --no-progress `
  --seed 103 *> artifacts\scheduled_runs\orientation_10epoch.log
```

Position and orientation are independent and can be run in parallel.  If CUDA
runs out of memory, lower both to `--batch-size 64`; if the GPU is underused and
memory is safe, try `--batch-size 256`.

### Scheduler status check

This read-only check is safe to run from Task Scheduler every 30-60 minutes:

```powershell
@'
import json, time
from pathlib import Path

checks = {
    "position": {
        "path": Path("models/position.metrics.jsonl"),
        "metric": ("val", "mean_radial_error_mm"),
        "warn_above": 0.5,
        "unit": "mm",
    },
    "orientation": {
        "path": Path("models/orientation.metrics.jsonl"),
        "metric": ("val", "mean_abs_error_deg"),
        "warn_above": 1.5,
        "unit": "deg",
    },
}

for name, cfg in checks.items():
    path = cfg["path"]
    if not path.exists():
        print(f"{name}: waiting for first epoch; no {path}")
        continue
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        print(f"{name}: metrics file exists but is empty")
        continue
    row = json.loads(lines[-1])
    section, key = cfg["metric"]
    value = row.get(section, {}).get(key)
    age_min = (time.time() - path.stat().st_mtime) / 60.0
    status = "OK"
    if value is None:
        status = "WARN missing metric"
    elif float(value) > cfg["warn_above"]:
        status = "WARN non-ideal"
    if age_min > 90:
        status = "WARN stale metrics"
    print(
        f"{name}: {status}; epoch={row.get('epoch')} best_epoch={row.get('best_epoch')} "
        f"{key}={value} {cfg['unit']} age={age_min:.1f}min"
    )
'@ | ..\.venv\Scripts\python.exe -
```

Suggested intervention thresholds:

```text
position: warn/stop if mean_radial_error_mm > 0.5 after epoch 1
orientation: warn/stop if mean_abs_error_deg > 1.5 after epoch 1
both: warn if metrics file has not updated for >90 minutes while the process is expected to be running
```

When a 10-epoch batch finishes cleanly and metrics are good, either stop there
or continue to the next batch with resume:

```powershell
..\.venv\Scripts\python.exe scripts\train_position.py `
  --dataset data\train_seen_120k_edge_fast `
  --val-dataset data\val_unseen_4k_edge_orientable `
  --out models\position.pt `
  --epochs 20 `
  --batch-size 128 `
  --lr 0.0003 `
  --base-channels 32 `
  --device cuda `
  --amp `
  --limit 4096 `
  --val-limit 1024 `
  --limit-strategy stratified `
  --resume models\position.last.pt `
  --patience 3 `
  --no-progress `
  --seed 102
```

Remember: `--epochs` is the total target epoch count.  Resuming from epoch 10
with `--epochs 20` runs epochs 11-20.

## Notes

- The dataset collector now prints progress immediately and after every
  `--progress-every` samples.  It also streams class counts instead of keeping
  all masks in RAM, so large dataset collection should not silently balloon
  memory.
- If `tqdm` is not installed, training now prints plain stdout batch progress
  instead of looking stuck until the end of the epoch.  Install `tqdm` only if
  you want nicer progress bars.
- Use `--device cuda` if this environment has a CUDA-enabled PyTorch install.
  Check with:

```powershell
@'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
'@ | ..\.venv\Scripts\python.exe -
```

  If it prints `+cpu` or `False`, install a CUDA PyTorch wheel.  Use the
  official PyTorch selector for your exact driver/CUDA combination:

```text
https://pytorch.org/get-started/locally/
```

  Typical Windows pip command for a modern NVIDIA GPU is one of:

```powershell
..\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
..\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

  If `cu128` is not compatible with your installed NVIDIA driver, use the
  install selector and pick a lower CUDA wheel such as `cu126` or `cu118`.
- Use `--no-progress` for CI/log files; default terminal runs show tqdm bars
  when `tqdm` is installed.
- The current renderer is still synthetic/clean.  This pipeline trains the
  existing simulated perception stack; domain randomization and PyBullet
  segmentation-ID rendering remain future realism upgrades.




