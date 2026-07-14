#!/usr/bin/env python
"""Select an uncertainty/diversity-balanced frame batch from saved predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.sim2real.active_learning import (  # noqa: E402
    entropy_uncertainty,
    margin_uncertainty,
    select_active_learning_indices,
)


def _load(path: Path, method: str):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        frame_ids = data.get("frame_ids", data.get("files"))
        features = data.get("embeddings", data.get("features"))
        if "uncertainty" in data:
            uncertainty = np.asarray(data["uncertainty"], dtype=float).reshape(-1)
        else:
            probabilities = data.get("probabilities", data.get("probs"))
            if probabilities is None and "logits" in data:
                logits = data["logits"].astype(np.float64)
                logits -= logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            if probabilities is None:
                raise ValueError("NPZ needs uncertainty, probabilities/probs, or logits")
            scorer = entropy_uncertainty if method == "entropy" else margin_uncertainty
            uncertainty = scorer(probabilities)
        ids = (
            [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in frame_ids.tolist()]
            if frame_ids is not None
            else [str(i) for i in range(len(uncertainty))]
        )
        return ids, uncertainty, features
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("items", payload) if isinstance(payload, dict) else payload
    ids = [str(item.get("frame", item.get("image", item.get("id", index)))) for index, item in enumerate(records)]
    uncertainty = np.asarray([item["uncertainty"] for item in records], dtype=float)
    features = None
    if records and all("embedding" in item for item in records):
        features = np.asarray([item["embedding"] for item in records], dtype=float)
    return ids, uncertainty, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="NPZ or JSON prediction records")
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--method", choices=["entropy", "margin"], default="entropy")
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame_ids, uncertainty, features = _load(args.predictions, args.method)
    indices = select_active_learning_indices(
        uncertainty,
        args.budget,
        features=features,
        diversity_weight=args.diversity_weight,
        seed=args.seed,
    )
    payload = {
        "schema": "sfn.active_learning_selection",
        "schema_version": 1,
        "source": str(args.predictions),
        "seed": args.seed,
        "method": args.method,
        "diversity_weight": args.diversity_weight,
        "budget": args.budget,
        "items": [
            {
                "selection_rank": rank,
                "source_index": index,
                "frame": frame_ids[index],
                "uncertainty": float(uncertainty[index]),
            }
            for rank, index in enumerate(indices, 1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": len(indices)}, sort_keys=True))


if __name__ == "__main__":
    main()
