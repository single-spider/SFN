#!/usr/bin/env python
"""Rectify a Panda-native dataset to a metric top-down task-plane view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.data.dataset import NPZDataset  # noqa: E402
from sfn.panda.camera_canonicalization import CanonicalCamera, PandaCameraCanonicalizer  # noqa: E402
from sfn.panda.config import PandaConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--resolution", type=int, default=500)
    ap.add_argument("--extent-mm", type=float, default=50.0)
    args = ap.parse_args()
    source, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    pc = manifest["panda_config"]
    cc = manifest["camera_config"]
    panda = PandaConfig(**{k: v for k, v in pc.items() if k in PandaConfig.__dataclass_fields__})
    canonical = CanonicalCamera(args.resolution, args.resolution, args.extent_mm, args.extent_mm)
    warp = PandaCameraCanonicalizer(
        panda, int(cc["crop_width"]), int(cc["crop_height"]), float(cc["fov_y_deg"]), canonical
    )
    ds = NPZDataset(source)
    chunks = []
    keys = ["rgb", "mask", "pose_error", "position_target", "orientation_index", "shape_id", "sample_id", "seed"]
    for ci, start in enumerate(range(0, len(ds), args.chunk_size)):
        rows = [ds[i] for i in range(start, min(len(ds), start + args.chunk_size))]
        arrays = {k: np.stack([r[k] for r in rows]) for k in keys}
        arrays["camera_variant"] = np.asarray([r.get("camera_variant", 0) for r in rows], dtype=np.int16)
        arrays["augmentation_json"] = np.asarray(
            [
                r.get("augmentation_json", json.dumps({"version": 1, "level": "none", "seed": None}))
                for r in rows
            ]
        )
        arrays["rgb"] = np.stack([warp.warp_rgb(r["rgb"]) for r in rows])
        arrays["mask"] = np.stack([warp.warp_mask(r["mask"]) for r in rows])
        path = out / f"{manifest['split']}_{ci:03d}.npz"
        np.savez_compressed(path, **arrays)
        chunks.append(
            {"path": path.name, "samples": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    result = {
        **manifest,
        "schema_version": 2,
        "schema_revision": 0,
        "metadata_path": "manifest.json",
        "randomization": manifest.get(
            "randomization",
            {
                "implementation": "none",
                "record_version": 1,
                "level": "none",
                "per_sample_field": "augmentation_json",
            },
        ),
        "dataset_type": "panda_native_camera_canonical_metric",
        "samples": len(ds),
        "chunks": chunks,
        "canonical_camera": canonical.__dict__,
        "source_dataset": str(source),
        "homography": warp.matrix.tolist(),
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "samples": len(ds), "chunks": len(chunks)}, indent=2))


if __name__ == "__main__":
    main()
