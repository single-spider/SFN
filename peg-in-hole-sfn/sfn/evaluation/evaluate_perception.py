"""Perception checkpoint evaluation for segmentation, position, orientation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data.dataset import NPZDataset
from ..geometry import decode_orientation, decode_position
from ..training.common import load_checkpoint_cpu


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for perception evaluation.") from exc
    return torch


def _load_model(task: str, checkpoint: str | Path):
    _require_torch()
    from ..models.orientation import orientation_model_from_config
    from ..models.position import CalibratedGeometricPositionNet, PositionNet
    from ..models.segmentation import SegmentationModel

    ckpt = load_checkpoint_cpu(checkpoint)
    if task == "segmentation":
        model = SegmentationModel(
            **{k: v for k, v in ckpt.get("model_config", {}).items() if k in {"in_channels", "classes", "base"}}
        )
    elif task == "position":
        cfg = ckpt.get("model_config", {})
        if cfg.get("model_type") == "calibrated_geometric":
            model = CalibratedGeometricPositionNet(
                feature_mean=cfg["feature_mean"],
                feature_std=cfg["feature_std"],
                weights=cfg["weights"],
                grid_size=cfg.get("grid_size", 21),
                temperature_mm=cfg.get("temperature_mm", 0.5),
            )
        else:
            model = PositionNet(**{k: v for k, v in cfg.items() if k in {"in_channels", "grid_size", "base"}})
    elif task == "orientation":
        cfg = ckpt.get("model_config", {})
        model = orientation_model_from_config(cfg)
    else:
        raise ValueError(f"Unknown task {task}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _shape_id(item) -> str:
    return str(item.get("shape_id", "unknown"))


def _position_summary(rows: list[tuple[int, int, int, int]]) -> dict:
    exact = within1 = within2 = within5 = 0
    abs_x = []
    abs_y = []
    radial = []
    for pr, pc, gr, gc in rows:
        dr = abs(pr - gr)
        dc = abs(pc - gc)
        cell_dist = max(dr, dc)
        exact += int(cell_dist == 0)
        within1 += int(cell_dist <= 1)
        within2 += int(cell_dist <= 2)
        within5 += int(cell_dist <= 5)
        pdx, pdy = decode_position(pr, pc)
        gdx, gdy = decode_position(gr, gc)
        ex = abs(pdx - gdx) * 1000.0
        ey = abs(pdy - gdy) * 1000.0
        abs_x.append(ex)
        abs_y.append(ey)
        radial.append(float(np.hypot(ex, ey)))
    n = max(1, len(rows))
    return {
        "samples": len(rows),
        "exact_cell_accuracy": exact / n,
        "within_1_cell_accuracy": within1 / n,
        "within_2_cell_accuracy": within2 / n,
        "within_5_cell_accuracy": within5 / n,
        "mean_abs_x_mm": float(np.mean(abs_x)) if abs_x else 0.0,
        "mean_abs_y_mm": float(np.mean(abs_y)) if abs_y else 0.0,
        "mean_radial_error_mm": float(np.mean(radial)) if radial else 0.0,
    }


def _continuous_position_summary(rows: list[tuple[float, float, float, float]]) -> dict:
    """Summarize metric XY estimates without quantizing them to 1 mm cells."""
    if not rows:
        return {
            "samples": 0,
            "mean_abs_x_mm": 0.0,
            "mean_abs_y_mm": 0.0,
            "mean_radial_error_mm": 0.0,
            "median_radial_error_mm": 0.0,
            "p95_radial_error_mm": 0.0,
            "within_0_5_mm": 0.0,
            "within_1_mm": 0.0,
            "within_2_mm": 0.0,
        }
    arr = np.asarray(rows, dtype=np.float64)
    err = (arr[:, :2] - arr[:, 2:]) * 1000.0
    radial = np.linalg.norm(err, axis=1)
    return {
        "samples": len(rows),
        "mean_abs_x_mm": float(np.mean(np.abs(err[:, 0]))),
        "mean_abs_y_mm": float(np.mean(np.abs(err[:, 1]))),
        "mean_radial_error_mm": float(np.mean(radial)),
        "median_radial_error_mm": float(np.median(radial)),
        "p95_radial_error_mm": float(np.percentile(radial, 95)),
        "within_0_5_mm": float(np.mean(radial <= 0.5)),
        "within_1_mm": float(np.mean(radial <= 1.0)),
        "within_2_mm": float(np.mean(radial <= 2.0)),
    }


def _orientation_summary(rows: list[tuple[int, int, list[float]]]) -> dict:
    exact = within2 = within4 = 0
    abs_err = []
    confusion = np.zeros((11, 11), dtype=np.int64)
    for pred_idx, gt_idx, angles in rows:
        pred = decode_orientation(pred_idx, angles)
        gt = decode_orientation(gt_idx, angles)
        err = abs(pred - gt)
        exact += int(pred_idx == gt_idx)
        within2 += int(err <= 2.0)
        within4 += int(err <= 4.0)
        abs_err.append(err)
        if gt_idx < confusion.shape[0] and pred_idx < confusion.shape[1]:
            confusion[gt_idx, pred_idx] += 1
    n = max(1, len(rows))
    return {
        "samples": len(rows),
        "exact_candidate_accuracy": exact / n,
        "within_2_deg_accuracy": within2 / n,
        "within_4_deg_accuracy": within4 / n,
        "mean_abs_error_deg": float(np.mean(abs_err)) if abs_err else 0.0,
        "confusion": confusion.tolist(),
    }


def evaluate_segmentation(
    dataset: str | Path, checkpoint: str | Path, limit: int | None = None, per_shape: bool = False
) -> dict:
    torch = _require_torch()
    ds = NPZDataset(dataset)
    model = _load_model("segmentation", checkpoint)
    confusion = np.zeros((3, 3), dtype=np.int64)
    n = min(len(ds), limit) if limit else len(ds)
    by_shape_confusion: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            x = torch.as_tensor(item["rgb"][None], dtype=torch.float32) / 255.0
            pred = torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.int64)
            gt = item["mask"].astype(np.int64)
            local = np.zeros((3, 3), dtype=np.int64)
            for g, p in zip(gt.reshape(-1), pred.reshape(-1), strict=False):
                if 0 <= g < 3 and 0 <= p < 3:
                    local[g, p] += 1
            confusion += local
            if per_shape:
                by_shape_confusion[_shape_id(item)] += local
    ious = []
    for c in range(3):
        tp = confusion[c, c]
        denom = confusion[c, :].sum() + confusion[:, c].sum() - tp
        ious.append(float(tp / denom) if denom else 1.0)
    result = {
        "task": "segmentation",
        "samples": n,
        "pixel_accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "mean_iou": float(np.mean(ious)),
        "class_iou": {str(i): ious[i] for i in range(3)},
        "confusion": confusion.tolist(),
    }
    if per_shape:
        result["per_shape"] = {}
        for shape, shape_confusion in sorted(by_shape_confusion.items()):
            shape_ious = []
            for c in range(3):
                tp = shape_confusion[c, c]
                denom = shape_confusion[c, :].sum() + shape_confusion[:, c].sum() - tp
                shape_ious.append(float(tp / denom) if denom else 1.0)
            result["per_shape"][shape] = {
                "pixel_accuracy": float(np.trace(shape_confusion) / max(1, shape_confusion.sum())),
                "mean_iou": float(np.mean(shape_ious)),
                "class_iou": {str(i): shape_ious[i] for i in range(3)},
                "confusion": shape_confusion.tolist(),
            }
    return result


def evaluate_position(
    dataset: str | Path, checkpoint: str | Path, limit: int | None = None, per_shape: bool = False
) -> dict:
    torch = _require_torch()
    ds = NPZDataset(dataset)
    model = _load_model("position", checkpoint)
    n = min(len(ds), limit) if limit else len(ds)
    rows: list[tuple[int, int, int, int]] = []
    continuous_rows: list[tuple[float, float, float, float]] = []
    by_shape: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            mask = torch.as_tensor(item["mask"][None, None], dtype=torch.float32) / 2.0
            logits = model(mask)
            if logits.dim() == 2:
                prob = torch.softmax(logits, dim=1)[0]
                pred_flat = int(torch.argmax(prob).cpu())
                grid = int(round(logits.shape[1] ** 0.5))
            else:
                prob = torch.softmax(logits, dim=1)[0, 1]
                pred_flat = int(torch.argmax(prob).cpu())
                grid = prob.shape[1]
            pr, pc = divmod(pred_flat, grid)
            gt_flat = int(np.argmax(item["position_target"]))
            gr, gc = divmod(gt_flat, item["position_target"].shape[1])
            row = (pr, pc, gr, gc)
            rows.append(row)
            if hasattr(model, "predict_continuous"):
                xy = model.predict_continuous(mask)[0].cpu().numpy()
                truth = np.asarray(item["pose_error"], dtype=np.float64)[:2]
                continuous_rows.append((float(xy[0]), float(xy[1]), float(truth[0]), float(truth[1])))
            if per_shape:
                by_shape[_shape_id(item)].append(row)
    result = {
        "task": "position",
        **_position_summary(rows),
    }
    if continuous_rows:
        result["continuous_metric"] = _continuous_position_summary(continuous_rows)
    if per_shape:
        result["per_shape"] = {shape: _position_summary(shape_rows) for shape, shape_rows in sorted(by_shape.items())}
    return result


def evaluate_orientation(
    dataset: str | Path, checkpoint: str | Path, limit: int | None = None, per_shape: bool = False
) -> dict:
    torch = _require_torch()
    ds = NPZDataset(dataset)
    model = _load_model("orientation", checkpoint)
    n = min(len(ds), limit) if limit else len(ds)
    rows: list[tuple[int, int, list[float]]] = []
    by_shape: dict[str, list[tuple[int, int, list[float]]]] = defaultdict(list)
    angles = getattr(model, "angles", [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            mask = torch.as_tensor(item["mask"][None, None], dtype=torch.float32) / 2.0
            pred_idx = int(torch.argmax(model(mask), dim=1)[0].cpu())
            gt_idx = int(item["orientation_index"])
            row = (pred_idx, gt_idx, angles)
            rows.append(row)
            if per_shape:
                by_shape[_shape_id(item)].append(row)
    result = {
        "task": "orientation",
        **_orientation_summary(rows),
    }
    if per_shape:
        result["per_shape"] = {
            shape: _orientation_summary(shape_rows) for shape, shape_rows in sorted(by_shape.items())
        }
    return result


def evaluate_predicted_mask_cascade(
    dataset: str | Path,
    segmentation: str | Path,
    position: str | Path,
    orientation: str | Path,
    limit: int | None = None,
    per_shape: bool = False,
) -> dict:
    """Evaluate the deployed RGB -> predicted mask -> XY/yaw path."""
    torch = _require_torch()
    ds = NPZDataset(dataset)
    seg_model = _load_model("segmentation", segmentation)
    pos_model = _load_model("position", position)
    ori_model = _load_model("orientation", orientation)
    angles = getattr(ori_model, "angles", [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    n = min(len(ds), limit) if limit else len(ds)
    pos_rows = []
    continuous_pos_rows = []
    ori_rows = []
    pos_by_shape = defaultdict(list)
    ori_by_shape = defaultdict(list)
    invalid = 0
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            rgb = torch.as_tensor(item["rgb"][None], dtype=torch.float32) / 255.0
            pred_mask = torch.argmax(seg_model(rgb), dim=1)
            invalid += int(not bool(torch.any(pred_mask == 1)) or not bool(torch.any(pred_mask == 2)))
            encoded = pred_mask[:, None].float() / 2.0
            pos_logits = pos_model(encoded)
            pred_flat = int(torch.argmax(pos_logits.reshape(1, -1), dim=1)[0])
            grid = int(round(pos_logits.reshape(1, -1).shape[1] ** 0.5))
            pr, pc = divmod(pred_flat, grid)
            gt_flat = int(np.argmax(item["position_target"]))
            gr, gc = divmod(gt_flat, item["position_target"].shape[1])
            prow = (pr, pc, gr, gc)
            if hasattr(pos_model, "predict_continuous"):
                xy = pos_model.predict_continuous(encoded)[0].cpu().numpy()
                truth = np.asarray(item["pose_error"], dtype=np.float64)[:2]
                continuous_pos_rows.append((float(xy[0]), float(xy[1]), float(truth[0]), float(truth[1])))
            orow = (int(torch.argmax(ori_model(encoded), dim=1)[0]), int(item["orientation_index"]), angles)
            pos_rows.append(prow)
            ori_rows.append(orow)
            if per_shape:
                shape = _shape_id(item)
                pos_by_shape[shape].append(prow)
                ori_by_shape[shape].append(orow)
    result = {
        "task": "predicted_mask_cascade",
        "samples": n,
        "invalid_mask_rate": invalid / max(1, n),
        "position": _position_summary(pos_rows),
        "orientation": _orientation_summary(ori_rows),
    }
    if continuous_pos_rows:
        result["position_continuous_metric"] = _continuous_position_summary(continuous_pos_rows)
    if per_shape:
        result["per_shape"] = {
            shape: {
                "position": _position_summary(pos_by_shape[shape]),
                "orientation": _orientation_summary(ori_by_shape[shape]),
            }
            for shape in sorted(pos_by_shape)
        }
    return result


def evaluate_all(
    dataset: str | Path,
    segmentation=None,
    position=None,
    orientation=None,
    limit: int | None = None,
    per_shape: bool = False,
) -> dict:
    out = {}
    if segmentation:
        out["segmentation"] = evaluate_segmentation(dataset, segmentation, limit, per_shape=per_shape)
    if position:
        out["position"] = evaluate_position(dataset, position, limit, per_shape=per_shape)
    if orientation:
        out["orientation"] = evaluate_orientation(dataset, orientation, limit, per_shape=per_shape)
    if segmentation and position and orientation:
        out["predicted_mask_cascade"] = evaluate_predicted_mask_cascade(
            dataset, segmentation, position, orientation, limit, per_shape=per_shape
        )
    return out


def write_metrics(metrics: dict, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
