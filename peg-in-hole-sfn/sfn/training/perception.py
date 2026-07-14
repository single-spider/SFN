"""Robust supervised training for SFN perception models.

The old function in this file was intentionally tiny.  This version keeps the
same public ``train_perception(...)`` entry point while adding the pieces needed
for real terminal runs:

* progress bars / useful epoch logs,
* train/validation metrics,
* checkpoint-last and checkpoint-best files,
* resume from interrupted runs,
* early stopping,
* class-balanced/focal losses,
* lightweight hyper-parameter grid search.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..data.dataset import NPZDataset
from ..geometry import decode_orientation, decode_position
from ..training.common import load_checkpoint_cpu, make_checkpoint, save_checkpoint

TaskName = Literal["segmentation", "position", "orientation"]


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset, Subset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for perception training. Install torch in the active environment."
        ) from exc
    return torch, nn, DataLoader, Dataset, Subset


@dataclass
class PerceptionTrainOptions:
    device: str = "auto"
    val_dataset: str | Path | None = None
    val_fraction: float = 0.0
    val_limit: int | None = None
    num_workers: int = 0
    base_channels: int = 16
    weight_decay: float = 1e-4
    loss: str = "auto"  # auto, ce, weighted_ce, focal
    class_weight: str = "median"  # none, median, inverse, effective
    position_pos_weight: float = 25.0
    focal_gamma: float = 2.0
    resume: str | Path | None = None
    patience: int | None = None
    min_delta: float = 0.0
    checkpoint_every: int = 1
    keep_epoch_checkpoints: bool = False
    amp: bool = False
    progress: bool = True
    metric: str | None = None
    limit_strategy: str = "stratified"  # prefix, shuffle, stratified
    orientation_architecture: str = "relative"


class TorchNPZDataset:
    def __init__(self, base: NPZDataset, task: TaskName, indices: list[int] | None = None, limit: int | None = None):
        self.base = base
        self.task = task
        all_indices = list(range(len(base))) if indices is None else list(indices)
        if limit is not None:
            all_indices = all_indices[: int(limit)]
        self.indices = all_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        torch, _nn, _dl, _ds, _subset = _require_torch()
        item = self.base[self.indices[int(idx)]]
        mask = torch.as_tensor(item["mask"], dtype=torch.long)
        if self.task == "segmentation":
            x = torch.as_tensor(item["rgb"], dtype=torch.float32) / 255.0
            y = mask
        elif self.task == "position":
            x = (mask.float() / 2.0).unsqueeze(0)
            # One and only one x-y offset cell is correct.  Train it as a
            # 441-way classifier instead of a heavily imbalanced 21x21 binary
            # segmentation map.
            y = torch.as_tensor(int(np.argmax(item["position_target"].reshape(-1))), dtype=torch.long)
        else:
            x = (mask.float() / 2.0).unsqueeze(0)
            y = torch.as_tensor(item["orientation_index"], dtype=torch.long)
        return x, y


class FocalLoss:
    def __init__(self, weight=None, gamma: float = 2.0):
        self.weight = weight
        self.gamma = float(gamma)

    def to(self, dev):
        if self.weight is not None:
            self.weight = self.weight.to(dev)
        return self

    def __call__(self, logits, target):
        torch, nn, *_ = _require_torch()
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce.detach())
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def _device(name: str):
    torch, *_ = _require_torch()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, leave=False)
    except ModuleNotFoundError:
        return _PlainProgress(iterable, desc=desc)


class _PlainProgress:
    """Tiny stdout progress fallback when tqdm is not installed.

    Several user terminals in this project have no ``tqdm`` package installed;
    without a fallback a large CUDA epoch can look frozen for many minutes.
    This wrapper prints coarse batch progress with no extra dependency.
    """

    def __init__(self, iterable, *, desc: str):
        self.iterable = iterable
        self.desc = desc
        self.index = 0
        self.postfix: dict[str, Any] = {}
        try:
            self.total = len(iterable)
        except TypeError:
            self.total = None
        self.started = time.time()
        if self.total:
            self.every = max(1, min(100, self.total // 20 or 1))
            print(f"[{self.desc}] starting {self.total} batches", flush=True)
        else:
            self.every = 100
            print(f"[{self.desc}] starting", flush=True)

    def __iter__(self):
        for item in self.iterable:
            self.index += 1
            yield item
            if self.index == 1 or self.index % self.every == 0 or (self.total and self.index == self.total):
                self._print()

    def set_postfix(self, **kwargs) -> None:
        self.postfix.update(kwargs)

    def _print(self) -> None:
        elapsed = max(1e-6, time.time() - self.started)
        rate = self.index / elapsed
        if self.total:
            pct = 100.0 * self.index / max(1, self.total)
            head = f"[{self.desc}] {self.index}/{self.total} batches ({pct:.1f}%, {rate:.2f} batch/s)"
        else:
            head = f"[{self.desc}] {self.index} batches ({rate:.2f} batch/s)"
        if self.postfix:
            tail = " ".join(f"{k}={v}" for k, v in self.postfix.items())
            head = f"{head} {tail}"
        print(head, flush=True)


def _model_for_task(task: TaskName, base_channels: int, orientation_architecture: str = "relative"):
    from ..models.orientation import OrientationNet, RelativeOrientationNet
    from ..models.position import PositionNet
    from ..models.segmentation import SegmentationModel

    if task == "segmentation":
        model = SegmentationModel(base=base_channels)
        cfg = {"in_channels": 3, "classes": 3, "base": base_channels}
    elif task == "position":
        model = PositionNet(base=base_channels)
        cfg = {"in_channels": 1, "grid_size": 21, "base": base_channels}
    else:
        if orientation_architecture == "relative":
            model = RelativeOrientationNet(base=base_channels)
            model_type = RelativeOrientationNet.model_type
        elif orientation_architecture == "legacy":
            model = OrientationNet(base=base_channels)
            model_type = "legacy_joint_geometric_residual"
        else:
            raise ValueError("orientation_architecture must be one of: relative, legacy")
        cfg = {
            "in_channels": 1,
            "angles": list(model.angles),
            "base": base_channels,
            "model_type": model_type,
        }
    return model, cfg


def _class_counts(base: NPZDataset, task: TaskName) -> list[float] | None:
    manifest = getattr(base, "manifest", None) or {}
    if task == "segmentation" and isinstance(manifest.get("class_pixel_counts"), dict):
        counts = [float(manifest["class_pixel_counts"].get(str(i), 0.0)) for i in range(3)]
        return counts if all(c > 0 for c in counts) else None
    return None


def _weights_from_counts(counts: list[float] | None, strategy: str):
    torch, *_ = _require_torch()
    if not counts or strategy == "none":
        return None
    arr = np.asarray(counts, dtype=np.float64)
    if strategy == "inverse":
        weights = arr.sum() / np.maximum(arr, 1.0)
    elif strategy == "effective":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - np.power(beta, arr))
    else:  # median
        weights = np.median(arr) / np.maximum(arr, 1.0)
    weights = np.clip(weights, 1e-3, 50.0)
    weights = weights / weights.min()
    return torch.tensor(weights, dtype=torch.float32)


def _criterion(task: TaskName, base: NPZDataset, opts: PerceptionTrainOptions, dev):
    torch, nn, *_ = _require_torch()
    loss_name = opts.loss
    if loss_name == "auto":
        loss_name = "weighted_ce" if task in {"segmentation", "position"} else "ce"

    weight = None
    if task == "segmentation" and loss_name in {"weighted_ce", "focal"}:
        weight = _weights_from_counts(_class_counts(base, task), opts.class_weight)
    elif task == "position" and loss_name == "focal":
        # Multiclass position labels are not background-dominated, so no
        # positive-vs-negative class weighting is needed.
        weight = None
    if weight is not None:
        weight = weight.to(dev)

    if loss_name == "focal":
        return FocalLoss(weight=weight, gamma=opts.focal_gamma).to(dev)
    return nn.CrossEntropyLoss(weight=weight)


def _split_indices(indices_or_n, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(int(indices_or_n))) if isinstance(indices_or_n, int) else list(indices_or_n)
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    n = len(indices)
    val_n = int(round(n * float(val_fraction)))
    val_n = max(0, min(n - 1, val_n)) if n > 1 else 0
    return indices[val_n:], indices[:val_n]


def _target_key_for_chunk(task: TaskName, arrays, local_idx: int, position_flats=None) -> int:
    if task == "position":
        if position_flats is None:
            return int(np.argmax(arrays["position_target"][local_idx].reshape(-1)))
        return int(position_flats[local_idx])
    if task == "orientation":
        return int(arrays["orientation_index"][local_idx])
    return 0


def _round_robin_by_target(records: list[tuple[int, int]], limit: int, seed: int) -> list[int]:
    """Return indices balanced across target bins.

    ``records`` contains ``(global_index, target_key)`` pairs from one shape.
    """

    rng = random.Random(int(seed))
    buckets: dict[int, list[int]] = defaultdict(list)
    for global_idx, target_key in records:
        buckets[int(target_key)].append(int(global_idx))
    for values in buckets.values():
        rng.shuffle(values)
    keys = list(buckets)
    rng.shuffle(keys)
    chosen: list[int] = []
    while keys and len(chosen) < int(limit):
        next_keys: list[int] = []
        for key in keys:
            values = buckets[key]
            if values and len(chosen) < int(limit):
                chosen.append(values.pop())
            if values:
                next_keys.append(key)
        keys = next_keys
    return chosen


def _select_limited_indices(
    base: NPZDataset,
    task: TaskName,
    limit: int | None,
    seed: int,
    strategy: str = "stratified",
    candidates: list[int] | None = None,
) -> list[int] | None:
    """Select a deterministic subset without accidentally taking one shape prefix.

    Chunked datasets are written shape-by-shape.  A naive ``[:limit]`` subset can
    therefore train on a single shape and validate on a single unseen shape.
    ``stratified`` balances first by ``shape_id`` and then by task target bin.
    """

    if candidates is None:
        candidate_set = None
        candidate_count = len(base)
    else:
        candidate_set = set(int(i) for i in candidates)
        candidate_count = len(candidate_set)
    if limit is None or int(limit) >= candidate_count:
        if candidates is None:
            return None
        return list(candidates)
    limit = int(limit)
    if limit <= 0:
        return []

    strategy = (strategy or "stratified").lower()
    if strategy not in {"prefix", "shuffle", "stratified"}:
        raise ValueError("limit_strategy must be one of: prefix, shuffle, stratified")

    all_candidates = list(range(len(base))) if candidates is None else list(candidates)
    if strategy == "prefix":
        return all_candidates[:limit]
    rng = random.Random(int(seed))
    if strategy == "shuffle":
        rng.shuffle(all_candidates)
        return all_candidates[:limit]

    by_shape: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chunk_idx, arrays in enumerate(base._chunks):
        offset = int(base._offsets[chunk_idx])
        n = int(arrays["rgb"].shape[0])
        position_flats = None
        if task == "position":
            position_flats = np.argmax(arrays["position_target"].reshape(n, -1), axis=1)
        for local_idx in range(n):
            global_idx = offset + local_idx
            if candidate_set is not None and global_idx not in candidate_set:
                continue
            shape = str(arrays["shape_id"][local_idx])
            target_key = _target_key_for_chunk(task, arrays, local_idx, position_flats)
            by_shape[shape].append((global_idx, target_key))

    shapes = sorted(by_shape)
    rng.shuffle(shapes)
    if not shapes:
        return []
    quota, rem = divmod(limit, len(shapes))
    chosen: list[int] = []
    for i, shape in enumerate(shapes):
        shape_limit = quota + (1 if i < rem else 0)
        chosen.extend(_round_robin_by_target(by_shape[shape], shape_limit, seed + i + 17))

    # If some shapes had too few samples, top up from the remaining candidates.
    if len(chosen) < limit:
        chosen_set = set(chosen)
        remaining = [idx for idx in all_candidates if idx not in chosen_set]
        rng.shuffle(remaining)
        chosen.extend(remaining[: limit - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:limit]


def _make_loaders(
    task: TaskName,
    dataset_path: str | Path,
    batch_size: int,
    seed: int,
    limit: int | None,
    opts: PerceptionTrainOptions,
):
    torch, nn, DataLoader, Dataset, Subset = _require_torch()
    train_base = NPZDataset(dataset_path)
    train_indices: list[int] | None = None
    val_ds = None
    if opts.val_dataset is not None:
        val_base = NPZDataset(opts.val_dataset)
        val_indices = _select_limited_indices(val_base, task, opts.val_limit, seed + 1, opts.limit_strategy)
        val_ds = TorchNPZDataset(val_base, task, indices=val_indices, limit=None)
    elif opts.val_fraction > 0.0:
        base_indices = _select_limited_indices(train_base, task, limit, seed, opts.limit_strategy)
        train_indices, val_indices = _split_indices(
            base_indices if base_indices is not None else len(train_base), opts.val_fraction, seed
        )
        val_indices = _select_limited_indices(
            train_base, task, opts.val_limit, seed + 1, opts.limit_strategy, candidates=val_indices
        )
        val_ds = TorchNPZDataset(train_base, task, indices=val_indices, limit=None)
    else:
        train_indices = _select_limited_indices(train_base, task, limit, seed, opts.limit_strategy)
    train_ds = TorchNPZDataset(train_base, task, indices=train_indices, limit=None)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=False,
        generator=generator,
    )
    val_loader = None
    if val_ds is not None and len(val_ds):
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=False
        )
    return train_base, train_loader, val_loader


def _segmentation_metrics(confusion: np.ndarray) -> dict[str, Any]:
    ious = []
    for c in range(3):
        tp = confusion[c, c]
        denom = confusion[c, :].sum() + confusion[:, c].sum() - tp
        ious.append(float(tp / denom) if denom else 1.0)
    return {
        "pixel_accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "mean_iou": float(np.mean(ious)),
        "class_iou": {str(i): ious[i] for i in range(3)},
        "confusion": confusion.tolist(),
    }


def _metric_mode_for_name(metric: str) -> str:
    return "max" if any(k in metric for k in ["iou", "accuracy"]) else "min"


def _metric_spec(task: TaskName, metric: str | None = None) -> tuple[str, str]:
    if metric:
        return metric, _metric_mode_for_name(metric)
    return _default_metric(task)


def _move_optimizer_state_to_device(optimizer, dev) -> None:
    """PyTorch loads optimizer tensors onto CPU from CPU checkpoints.

    If the model is on CUDA and the Adam moments stay on CPU, the next
    optimizer step can fail with a device mismatch.  Keep this small and
    generic so resume works across CPU/GPU terminals.
    """

    torch, *_ = _require_torch()
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(dev)


def _retime_cosine_scheduler_after_resume(scheduler, optimizer, epochs: int, start_epoch: int) -> None:
    """Keep resumed cosine LR schedules sane when extending a run.

    A common terminal workflow is:

    ``--epochs 1`` smoke/interrupted run -> resume with ``--epochs 50``.

    The checkpoint stores the optimizer LR after the old scheduler stepped.  If
    the old horizon was one epoch, that LR is already zero; simply changing
    ``T_max`` would still train the first resumed epoch at zero LR.  Recompute
    the LR for ``start_epoch`` under the new total horizon so the next epoch
    continues on the requested schedule.
    """

    if not hasattr(scheduler, "T_max") or not hasattr(scheduler, "base_lrs"):
        return
    scheduler.T_max = max(1, int(epochs))
    t = max(0, min(int(start_epoch), int(scheduler.T_max)))
    eta_min = float(getattr(scheduler, "eta_min", 0.0))
    lrs = [
        eta_min + (float(base_lr) - eta_min) * (1.0 + math.cos(math.pi * t / float(scheduler.T_max))) / 2.0
        for base_lr in scheduler.base_lrs
    ]
    for group, lr_value in zip(optimizer.param_groups, lrs, strict=False):
        group["lr"] = lr_value
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = list(lrs)


def evaluate_epoch(
    task: TaskName, model, loader, criterion, dev, *, progress: bool = False, desc: str | None = None
) -> dict[str, Any]:
    torch, nn, *_ = _require_torch()
    model.eval()
    total_loss = 0.0
    total_batches = 0
    samples = 0
    confusion = np.zeros((3, 3), dtype=np.int64)
    pos_exact = pos_w1 = pos_w2 = pos_w5 = 0
    pos_abs_x: list[float] = []
    pos_abs_y: list[float] = []
    pos_radial: list[float] = []
    ori_exact = ori_w2 = ori_w4 = 0
    ori_abs: list[float] = []
    ori_confusion = np.zeros((11, 11), dtype=np.int64)
    angles = getattr(model, "angles", [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

    with torch.no_grad():
        epoch_iter = _progress(loader, enabled=progress, desc=desc or f"{task} validation")
        for x, y in epoch_iter:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            samples += int(x.shape[0])
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{(total_loss / max(1, total_batches)):.4f}")

            if task == "segmentation":
                pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
                gt = y.cpu().numpy().astype(np.int64)
                valid = (gt >= 0) & (gt < 3) & (pred >= 0) & (pred < 3)
                if np.any(valid):
                    confusion += np.bincount(3 * gt[valid].reshape(-1) + pred[valid].reshape(-1), minlength=9).reshape(
                        3, 3
                    )
            elif task == "position":
                if logits.dim() == 2:
                    pred_flat = torch.argmax(logits, dim=1).cpu().numpy()
                    gt_flat = y.cpu().numpy()
                    grid = int(round(logits.shape[1] ** 0.5))
                else:
                    # Backwards compatibility for old 2-class heatmap
                    # checkpoints/tests.
                    prob = torch.softmax(logits, dim=1)[:, 1]
                    pred_flat = torch.argmax(prob.flatten(1), dim=1).cpu().numpy()
                    gt_flat = torch.argmax(y.flatten(1), dim=1).cpu().numpy()
                    grid = int(logits.shape[-1])
                for pf, gf in zip(pred_flat, gt_flat, strict=False):
                    pr, pc = divmod(int(pf), grid)
                    gr, gc = divmod(int(gf), grid)
                    dr, dc = abs(pr - gr), abs(pc - gc)
                    cell = max(dr, dc)
                    pos_exact += int(cell == 0)
                    pos_w1 += int(cell <= 1)
                    pos_w2 += int(cell <= 2)
                    pos_w5 += int(cell <= 5)
                    pdx, pdy = decode_position(pr, pc, grid_size=grid)
                    gdx, gdy = decode_position(gr, gc, grid_size=grid)
                    ex = abs(pdx - gdx) * 1000.0
                    ey = abs(pdy - gdy) * 1000.0
                    pos_abs_x.append(float(ex))
                    pos_abs_y.append(float(ey))
                    pos_radial.append(float(math.hypot(ex, ey)))
            else:
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                gt = y.cpu().numpy()
                for pi, gi in zip(pred, gt, strict=False):
                    pi, gi = int(pi), int(gi)
                    pa = decode_orientation(pi, angles)
                    ga = decode_orientation(gi, angles)
                    err = abs(pa - ga)
                    ori_exact += int(pi == gi)
                    ori_w2 += int(err <= 2.0)
                    ori_w4 += int(err <= 4.0)
                    ori_abs.append(float(err))
                    if 0 <= gi < 11 and 0 <= pi < 11:
                        ori_confusion[gi, pi] += 1

    metrics: dict[str, Any] = {
        "loss": float(total_loss / max(1, total_batches)),
        "samples": samples,
    }
    if task == "segmentation":
        metrics.update(_segmentation_metrics(confusion))
    elif task == "position":
        n = max(1, samples)
        metrics.update(
            {
                "exact_cell_accuracy": pos_exact / n,
                "within_1_cell_accuracy": pos_w1 / n,
                "within_2_cell_accuracy": pos_w2 / n,
                "within_5_cell_accuracy": pos_w5 / n,
                "mean_abs_x_mm": float(np.mean(pos_abs_x)) if pos_abs_x else 0.0,
                "mean_abs_y_mm": float(np.mean(pos_abs_y)) if pos_abs_y else 0.0,
                "mean_radial_error_mm": float(np.mean(pos_radial)) if pos_radial else 0.0,
            }
        )
    else:
        n = max(1, samples)
        metrics.update(
            {
                "exact_candidate_accuracy": ori_exact / n,
                "within_2_deg_accuracy": ori_w2 / n,
                "within_4_deg_accuracy": ori_w4 / n,
                "mean_abs_error_deg": float(np.mean(ori_abs)) if ori_abs else 0.0,
                "confusion": ori_confusion.tolist(),
            }
        )
    return metrics


def _default_metric(task: TaskName) -> tuple[str, str]:
    if task == "segmentation":
        return "mean_iou", "max"
    if task == "position":
        return "mean_radial_error_mm", "min"
    return "mean_abs_error_deg", "min"


def _is_better(value: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    return value > best + min_delta if mode == "max" else value < best - min_delta


def _paths(out: str | Path) -> dict[str, Path]:
    out = Path(out)
    return {
        "best": out,
        "last": out.with_name(out.stem + ".last" + out.suffix),
        "metrics": out.with_name(out.stem + ".metrics.jsonl"),
        "summary": out.with_name(out.stem + ".summary.json"),
    }


def _save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    task: TaskName,
    model_config: dict,
    epoch: int,
    global_step: int,
    metrics: dict,
    data_split: dict,
):
    ckpt = make_checkpoint(
        model_name=type(model).__name__,
        model_config={**model_config, "task": task},
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer is not None else None,
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        epoch=int(epoch),
        global_step=int(global_step),
        metrics=metrics,
        data_split=data_split,
    )
    save_checkpoint(path, ckpt)


def train_perception(
    task: TaskName,
    dataset_path: str | Path,
    out: str | Path,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-3,
    seed: int = 1,
    limit: int | None = None,
    *,
    val_dataset: str | Path | None = None,
    val_fraction: float = 0.0,
    val_limit: int | None = None,
    device: str = "auto",
    num_workers: int = 0,
    base_channels: int = 16,
    weight_decay: float = 1e-4,
    loss: str = "auto",
    class_weight: str = "median",
    position_pos_weight: float = 25.0,
    focal_gamma: float = 2.0,
    resume: str | Path | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    checkpoint_every: int = 1,
    keep_epoch_checkpoints: bool = False,
    amp: bool = False,
    progress: bool = True,
    metric: str | None = None,
    limit_strategy: str = "stratified",
    orientation_architecture: str = "relative",
) -> dict:
    """Train segmentation/position/orientation with resumability and metrics."""

    torch, nn, DataLoader, Dataset, Subset = _require_torch()
    if task not in {"segmentation", "position", "orientation"}:
        raise ValueError(f"Unknown perception task: {task}")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    dev = _device(device)
    opts = PerceptionTrainOptions(
        device=device,
        val_dataset=val_dataset,
        val_fraction=val_fraction,
        val_limit=val_limit,
        num_workers=num_workers,
        base_channels=base_channels,
        weight_decay=weight_decay,
        loss=loss,
        class_weight=class_weight,
        position_pos_weight=position_pos_weight,
        focal_gamma=focal_gamma,
        resume=resume,
        patience=patience,
        min_delta=min_delta,
        checkpoint_every=checkpoint_every,
        keep_epoch_checkpoints=keep_epoch_checkpoints,
        amp=amp,
        progress=progress,
        metric=metric,
        limit_strategy=limit_strategy,
        orientation_architecture=orientation_architecture,
    )
    train_base, train_loader, val_loader = _make_loaders(task, dataset_path, batch_size, seed, limit, opts)
    model, model_config = _model_for_task(task, base_channels, orientation_architecture)
    model.to(dev)
    criterion = _criterion(task, train_base, opts, dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp and dev.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_value: float | None = None
    best_epoch = 0
    paths = _paths(out)
    paths["best"].parent.mkdir(parents=True, exist_ok=True)

    if resume:
        ckpt = load_checkpoint_cpu(resume)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, dev)
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        _retime_cosine_scheduler_after_resume(scheduler, optimizer, int(epochs), start_epoch)
        best_value = ckpt.get("metrics", {}).get("best_metric_value")
        best_epoch = int(ckpt.get("metrics", {}).get("best_epoch", start_epoch))
        if not paths["best"].exists():
            shutil.copyfile(resume, paths["best"])
        if not paths["last"].exists():
            shutil.copyfile(resume, paths["last"])

    requested_metric_name, requested_metric_mode = _metric_spec(task, metric)

    metrics_file = paths["metrics"]
    if not resume and metrics_file.exists():
        metrics_file.unlink()
    no_improve = 0
    last_epoch_metrics: dict[str, Any] = ckpt.get("metrics", {}) if resume else {}
    started = time.time()

    for epoch in range(start_epoch + 1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        epoch_iter = _progress(train_loader, enabled=progress, desc=f"{task} epoch {epoch}/{epochs}")
        for x, y in epoch_iter:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(amp and dev.type == "cuda")):
                logits = model(x)
                batch_loss = criterion(logits, y)
            if not torch.isfinite(batch_loss):
                raise RuntimeError(f"Non-finite {task} loss at epoch {epoch}")
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(batch_loss.detach().cpu()))
            global_step += 1
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{np.mean(losses):.4f}")
        scheduler.step()

        train_metrics = {"loss": float(np.mean(losses)) if losses else 0.0, "samples": len(train_loader.dataset)}
        val_metrics = (
            evaluate_epoch(
                task,
                model,
                val_loader,
                criterion,
                dev,
                progress=progress,
                desc=f"{task} validation {epoch}/{epochs}",
            )
            if val_loader is not None
            else train_metrics
        )
        metric_name, metric_mode = requested_metric_name, requested_metric_mode
        if metric_name not in val_metrics:
            metric_name, metric_mode = "loss", "min"
        current_value = float(val_metrics[metric_name])
        improved = _is_better(current_value, best_value, metric_mode, min_delta)
        if improved:
            best_value = current_value
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        last_epoch_metrics = {
            "task": task,
            "epoch": epoch,
            "global_step": global_step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
            "validation_source": "validation" if val_loader is not None else "train",
            "requested_selection_metric": requested_metric_name,
            "requested_selection_mode": requested_metric_mode,
            "selection_metric": metric_name,
            "selection_mode": metric_mode,
            "selection_value": current_value,
            "best_metric_value": best_value,
            "best_epoch": best_epoch,
            "improved": bool(improved),
            "elapsed_sec": time.time() - started,
        }
        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(last_epoch_metrics, sort_keys=True) + "\n")
        data_split = getattr(train_base, "manifest", None) or {"dataset": str(dataset_path)}
        _save_training_checkpoint(
            paths["last"],
            model,
            optimizer,
            scheduler,
            task,
            model_config,
            epoch,
            global_step,
            last_epoch_metrics,
            data_split,
        )
        if improved:
            shutil.copyfile(paths["last"], paths["best"])
        if keep_epoch_checkpoints and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            epoch_path = paths["best"].with_name(paths["best"].stem + f".epoch{epoch:04d}" + paths["best"].suffix)
            shutil.copyfile(paths["last"], epoch_path)

        print(
            f"[{task}] epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_{metric_name}={current_value:.4f} "
            f"best={best_value:.4f} epoch={best_epoch}"
        )

        if patience is not None and no_improve >= int(patience):
            print(f"[{task}] early stopping after {no_improve} non-improving epochs")
            break

    if not last_epoch_metrics:
        last_epoch_metrics = {
            "task": task,
            "epoch": start_epoch,
            "global_step": global_step,
            "train": {},
            "val": {},
            "requested_selection_metric": requested_metric_name,
            "requested_selection_mode": requested_metric_mode,
            "selection_metric": "loss",
            "selection_mode": "min",
            "selection_value": best_value,
            "best_metric_value": best_value,
            "best_epoch": best_epoch,
            "improved": False,
            "elapsed_sec": time.time() - started,
        }
    paths["summary"].write_text(json.dumps(last_epoch_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "checkpoint": str(paths["best"]),
        "last_checkpoint": str(paths["last"]),
        "metrics_jsonl": str(metrics_file),
        "summary": str(paths["summary"]),
        **last_epoch_metrics,
    }


def parse_search_grid(text: str | None) -> list[dict[str, Any]]:
    """Parse JSON or ``k=a,b;k2=c,d`` grids into trial dictionaries."""
    if not text:
        return [{}]
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = {}
        for part in text.split(";"):
            if not part.strip():
                continue
            key, values = part.split("=", 1)
            raw[key.strip()] = [v.strip() for v in values.split(",")]
    if not isinstance(raw, dict):
        raise ValueError("search grid must be a JSON object or k=a,b;k2=c,d")

    def coerce(v):
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        s = str(v)
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    keys = list(raw.keys())
    vals = [v if isinstance(v, list) else [v] for v in raw.values()]
    return [{k: coerce(v) for k, v in zip(keys, combo, strict=False)} for combo in product(*vals)]


def hyperparameter_search(
    task: TaskName,
    dataset_path: str | Path,
    out_dir: str | Path,
    search_grid: str,
    *,
    max_trials: int | None = None,
    seed: int = 1,
    **base_kwargs,
) -> dict[str, Any]:
    """Run a small serial grid/random search and rank by validation metric."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = parse_search_grid(search_grid)
    rng = random.Random(seed)
    rng.shuffle(trials)
    if max_trials is not None:
        trials = trials[: int(max_trials)]

    rows: list[dict[str, Any]] = []
    metric_name, metric_mode = _metric_spec(task, base_kwargs.get("metric"))
    for i, params in enumerate(trials):
        kwargs = dict(base_kwargs)
        kwargs.update(params)
        trial_out = out_dir / f"{task}_trial_{i:03d}.pt"
        result = train_perception(task, dataset_path, trial_out, seed=seed + i, **kwargs)
        value = result.get("selection_value")
        rows.append(
            {
                "trial": i,
                "checkpoint": str(result.get("checkpoint", trial_out)),
                "metric": result.get("selection_metric", metric_name),
                "mode": result.get("selection_mode", metric_mode),
                "value": value,
                **params,
            }
        )

    metric_mode = rows[0]["mode"] if rows else metric_mode
    reverse = metric_mode == "max"
    rows.sort(key=lambda r: float(r["value"]), reverse=reverse)
    with (out_dir / "search_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metric_name = rows[0]["metric"] if rows else metric_name
    summary = {
        "task": task,
        "metric": metric_name,
        "mode": metric_mode,
        "best": rows[0] if rows else None,
        "trials": rows,
    }
    (out_dir / "search_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
