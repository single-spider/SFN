#!/usr/bin/env python
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.panda import PandaPegInHoleAlignmentEnv
from sfn.panda.config import PandaConfig


def main():
    env = PandaPegInHoleAlignmentEnv(shapes=["square-concave1"], panda_config=PandaConfig(native_camera=True))
    try:
        obs, info = env.reset(seed=8, options={"pose_error": [0.001, -0.001, 1.0], "shape": "square-concave1"})
        values, counts = np.unique(obs["mask"], return_counts=True)
        counts_by_label = {int(v): int(c) for v, c in zip(values, counts, strict=False)}
        metrics = {
            "success": bool(counts_by_label.get(1, 0) > 0 and counts_by_label.get(2, 0) > 0),
            "rgb_shape": list(obs["rgb"].shape),
            "mask_shape": list(obs["mask"].shape),
            "mask_counts": counts_by_label,
            "shape": info["shape"],
            "native_camera": True,
        }
        print(json.dumps(metrics, indent=2))
        if not metrics["success"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
