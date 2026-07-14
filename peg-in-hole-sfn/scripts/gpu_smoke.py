#!/usr/bin/env python
"""Run a deterministic CPU/CUDA parity smoke for the selected VSN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sfn.models.vsn import VirtualSensorNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, default=Path("models/segmentation_mesh_v2_ce.pt"))
    parser.add_argument("--position", type=Path, default=Path("models/position_mesh_v2_geometric.pt"))
    parser.add_argument("--orientation", type=Path, default=Path("models/orientation_mesh_v2_symmetry_hybrid.pt"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; GPU smoke was not executed")

    paths = (args.segmentation, args.position, args.orientation)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing checkpoint(s): {', '.join(missing)}")

    torch.manual_seed(20260713)
    rgb = torch.randint(0, 256, (2, 3, 96, 120), dtype=torch.uint8)
    cpu_model = VirtualSensorNetwork.from_checkpoints(*paths).eval()
    gpu_model = VirtualSensorNetwork.from_checkpoints(*paths).to("cuda").eval()
    with torch.inference_mode():
        cpu = cpu_model(rgb=rgb)
        gpu = gpu_model(rgb=rgb.to("cuda"))

    comparisons = {}
    for name in ("mask_logits", "position_logits", "orientation_scores", "dxy_m", "dyaw_deg"):
        left = getattr(cpu, name)
        right = getattr(gpu, name).cpu()
        comparisons[name] = {
            "max_abs_difference": float((left - right).abs().max()),
            "close": bool(torch.allclose(left, right, rtol=1e-4, atol=1e-5)),
        }
    comparisons["mask"] = {"identical": bool(torch.equal(cpu.mask, gpu.mask.cpu()))}
    passed = all(row.get("close", row.get("identical", False)) for row in comparisons.values())
    report = {
        "passed": passed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "checkpoints": [str(path) for path in paths],
        "comparisons": comparisons,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
