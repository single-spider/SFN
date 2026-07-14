#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.data.dataset import NPZDataset


def _features(dataset: str | Path):
    ds = NPZDataset(dataset)
    x_rows = []
    yaw_rows = []
    y_xy = []
    y_yaw = []
    yy = xx = None
    skipped = 0
    for i in range(len(ds)):
        item = ds[i]
        mask = item["mask"]
        if yy is None:
            yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
        peg = mask == 1
        base = mask == 2
        if peg.sum() == 0 or base.sum() == 0:
            skipped += 1
            continue
        px = xx[peg].astype(float)
        py = yy[peg].astype(float)
        bx = xx[base].astype(float)
        by = yy[base].astype(float)
        pcx, pcy, bcx, bcy = px.mean(), py.mean(), bx.mean(), by.mean()
        x_rows.append([pcx, pcy, bcx, bcy, pcx - bcx, pcy - bcy])
        x0, y0 = px - pcx, py - pcy
        mu20 = (x0 * x0).mean()
        mu02 = (y0 * y0).mean()
        mu11 = (x0 * y0).mean()
        angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02) * 180.0 / np.pi
        angle = ((angle + 45.0) % 90.0) - 45.0
        yaw_rows.append([angle])
        y_xy.append(item["pose_error"][:2])
        y_yaw.append(float(item["pose_error"][2]))
    return np.asarray(x_rows), np.asarray(yaw_rows), np.asarray(y_xy), np.asarray(y_yaw), skipped


def _summary(err):
    return {
        "mean": float(np.mean(err)) if len(err) else 0.0,
        "p90": float(np.percentile(err, 90)) if len(err) else 0.0,
        "max": float(np.max(err)) if len(err) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze Panda native body-ID mask pose observability.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    x_train, yaw_train, yxy_train, yyaw_train, skipped_train = _features(args.train)
    x_val, yaw_val, yxy_val, yyaw_val, skipped_val = _features(args.val)
    a_train = np.c_[x_train, np.ones(len(x_train))]
    xy_w = np.linalg.lstsq(a_train, yxy_train, rcond=None)[0]
    yaw_a_train = np.c_[yaw_train, np.ones(len(yaw_train))]
    yaw_w = np.linalg.lstsq(yaw_a_train, yyaw_train, rcond=None)[0]

    result = {
        "train": {"samples": int(len(x_train)), "skipped": int(skipped_train)},
        "val": {"samples": int(len(x_val)), "skipped": int(skipped_val)},
        "xy_linear_weights": xy_w.tolist(),
        "yaw_linear_weights": yaw_w.tolist(),
    }
    for name, x, yaw_x, yxy, yyaw in [
        ("train", x_train, yaw_train, yxy_train, yyaw_train),
        ("val", x_val, yaw_val, yxy_val, yyaw_val),
    ]:
        xy_pred = np.c_[x, np.ones(len(x))] @ xy_w
        xy_err = np.linalg.norm(xy_pred - yxy, axis=1) * 1000.0
        yaw_pred = np.c_[yaw_x, np.ones(len(yaw_x))] @ yaw_w
        yaw_err = np.abs(yaw_pred - yyaw)
        result[name].update(
            {
                "xy_error_mm": _summary(xy_err),
                "yaw_error_deg": _summary(yaw_err),
                "within_2deg_yaw": float(np.mean(yaw_err <= 2.0)) if len(yaw_err) else 0.0,
            }
        )
    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
