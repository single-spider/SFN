#!/usr/bin/env python
"""Fit the portable sub-cell geometric XY estimator from a training split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.data.dataset import NPZDataset  # noqa: E402
from sfn.models.position import CalibratedGeometricPositionNet  # noqa: E402


def _dataset_hash(root: Path) -> str:
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing dataset manifest: {manifest}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grid-size", type=int, default=21)
    ap.add_argument("--temperature-mm", type=float, default=0.5)
    args = ap.parse_args()

    import torch

    dataset_root = Path(args.dataset)
    ds = NPZDataset(dataset_root)
    raw_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, len(ds), args.batch_size):
        items = [ds[i] for i in range(start, min(len(ds), start + args.batch_size))]
        masks = torch.as_tensor(np.stack([x["mask"] for x in items]), dtype=torch.long)
        h, w = masks.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
        )
        raw = CalibratedGeometricPositionNet._region_features(masks, yy, xx)
        raw_rows.append(raw.numpy().astype(np.float64))
        targets.append(np.stack([np.asarray(x["pose_error"], dtype=np.float64)[:2] * 1000.0 for x in items]))
    features = np.concatenate(raw_rows)
    target = np.concatenate(targets)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    z = (features - mean) / np.maximum(std, 1e-6)
    design = np.concatenate((z, z * z, np.ones((len(z), 1))), axis=1)
    reg = float(args.ridge) * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ target)
    pred = design @ weights
    radial = np.linalg.norm(pred - target, axis=1)

    model = CalibratedGeometricPositionNet(
        mean, std, weights, grid_size=args.grid_size, temperature_mm=args.temperature_mm
    )
    config = {
        "model_type": "calibrated_geometric",
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weights": weights.tolist(),
        "grid_size": int(args.grid_size),
        "temperature_mm": float(args.temperature_mm),
        "fit_dataset": str(dataset_root),
        "dataset_manifest_sha256": _dataset_hash(dataset_root),
        "ridge_lambda": float(args.ridge),
    }
    checkpoint = {
        "schema_version": 1,
        "task": "position",
        "model_config": config,
        "model_state_dict": model.state_dict(),
        "training": {
            "algorithm": "standardized_quadratic_ridge",
            "samples": len(ds),
            "train_mean_radial_error_mm": float(radial.mean()),
            "train_p95_radial_error_mm": float(np.percentile(radial, 95)),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "source_revision": _revision(),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    print(json.dumps({"checkpoint": str(out), **checkpoint["training"]}, indent=2))


if __name__ == "__main__":
    main()
