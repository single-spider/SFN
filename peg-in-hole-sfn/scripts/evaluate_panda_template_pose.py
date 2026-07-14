#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sfn.data.dataset import NPZDataset
from sfn.panda.config import PandaConfig
from sfn.panda.template_pose import PandaTopdownTemplatePoseEstimator


def main():
    ap = argparse.ArgumentParser(description="Evaluate calibrated asset-template Panda pose estimation.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--segmentation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", choices=["topdown", "oblique"], default="topdown")
    a = ap.parse_args()
    ds = NPZDataset(a.dataset)
    models = {}
    rows = []
    seg = None
    if a.segmentation:
        import torch
        from sfn.evaluation.evaluate_perception import _load_model

        seg = _load_model("segmentation", a.segmentation)
    eye, target = ((0, 0, 0.2), (0, 0, 0.03)) if a.camera == "topdown" else ((0.06, -0.10, 0.16), (0, 0, 0))
    cfg = PandaConfig(
        native_camera=True, mesh_derived_alignment_z=True, camera_eye_offset_m=eye, camera_target_offset_m=target
    )
    for i in range(len(ds)):
        item = ds[i]
        shape = str(item["shape_id"])
        if shape not in models:
            models[shape] = PandaTopdownTemplatePoseEstimator(shape, cfg)
        mask = item["mask"]
        if seg is not None:
            import torch

            with torch.no_grad():
                mask = torch.argmax(seg(torch.as_tensor(item["rgb"][None], dtype=torch.float32) / 255.0), 1)[0].numpy()
        xy, yaw, conf, valid = models[shape].estimate(mask)
        truth = np.asarray(item["pose_error"], dtype=float)
        rows.append(
            {
                "shape": shape,
                "valid": valid,
                "confidence": conf,
                "xy_error_mm": float(np.linalg.norm(xy - truth[:2]) * 1000),
                "yaw_error_deg": float(abs(yaw - truth[2])),
                "pred_xy_m": xy.tolist(),
                "pred_yaw_deg": yaw,
                "truth": truth.tolist(),
            }
        )
    by = defaultdict(list)
    for r in rows:
        by[r["shape"]].append(r)

    def summary(rs):
        return {
            "samples": len(rs),
            "valid_rate": sum(r["valid"] for r in rs) / len(rs),
            "mean_xy_error_mm": float(np.mean([r["xy_error_mm"] for r in rs])),
            "p95_xy_error_mm": float(np.percentile([r["xy_error_mm"] for r in rs], 95)),
            "mean_yaw_error_deg": float(np.mean([r["yaw_error_deg"] for r in rs])),
            "p95_yaw_error_deg": float(np.percentile([r["yaw_error_deg"] for r in rs], 95)),
        }

    report = {
        "mask_source": "predicted_rgb" if seg else "simulator_semantic",
        "summary": summary(rows),
        "per_shape": {k: summary(v) for k, v in by.items()},
        "rows": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "per_shape": report["per_shape"]}, indent=2))


if __name__ == "__main__":
    main()
