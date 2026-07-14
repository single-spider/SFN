#!/usr/bin/env python
"""Write a reproducibility inventory without modifying datasets/checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sfn.reproducibility import checkpoint_inventory, dataset_inventory, git_state, runtime_state


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(root / "artifacts" / "software_audit" / "inventory.json"))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema_version": 1,
        "generated_unix": time.time(),
        "project_root": str(root),
        "backend_boundaries": {
            "toy_direct": "legacy rectangle renderer; smoke/proof-of-concept only",
            "mesh_orthographic": "actual peg and hole-opening meshes; final Cartesian visual baseline",
            "panda_kinematic": "idealized IK/coordinate validation; reset-based and not dynamic tracking",
            "panda_dynamic": "motor/contact execution without post-command teleport; pending gate",
        },
        "git": git_state(root),
        "runtime": runtime_state(),
        "datasets": dataset_inventory(root / "data"),
        "checkpoints": checkpoint_inventory(root / "models"),
    }
    out.write_text(json.dumps(inventory, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out), "datasets": len(inventory["datasets"]), "checkpoints": len(inventory["checkpoints"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
