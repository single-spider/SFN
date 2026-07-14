#!/usr/bin/env python
"""One-command software benchmark: mesh controllers, physics, and reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.training.common import file_sha256, run_metadata

PY = sys.executable
RUN_COMMANDS: list[list[str]] = []


def run(args, dry=False):
    RUN_COMMANDS.append([str(value) for value in args])
    print("==>", subprocess.list2cmdline([str(x) for x in args]), flush=True)
    if not dry:
        subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-panda", action="store_true")
    ap.add_argument("--panda-episodes", type=int, default=None, help="Episodes per Panda matrix cell (default: 40).")
    ap.add_argument("--out", default="artifacts/final_benchmark")
    a = ap.parse_args()
    episodes = 10 if a.quick else 40
    out = Path(a.out)
    started = time.time()
    base = [
        PY,
        "scripts/evaluate.py",
        "--config",
        "configs/mesh_insertion_tight.yaml",
        "--method",
        "all",
        "--mask_source",
        "predicted",
        "--task",
        "insertion",
        "--episodes",
        episodes,
        "--split",
        "test_unseen",
        "--segmentation",
        "models/segmentation_mesh_v2_ce.pt",
        "--position",
        "models/position_mesh_v2_geometric.pt",
        "--orientation",
        "models/orientation_mesh_v2_symmetry_hybrid.pt",
        "--sfms-policy",
        "models/sfms_mesh_v2_rl_best_compatible.pt",
        "--mfms-policy",
        "models/mfms_mesh_v2_teacher_compatible.pt",
    ]
    run([*base, "--robustness-profile", "clean", "--seed", 3100, "--out", out / "mesh_clean"], a.dry_run)
    run([*base, "--robustness-profile", "severity_4", "--seed", 3100, "--out", out / "mesh_severity4"], a.dry_run)
    run(
        [
            PY,
            "scripts/validate_standalone_physical_insertion.py",
            "--shapes",
            "all",
            "--out",
            out / "standalone_exact.json",
        ],
        a.dry_run,
    )
    run(
        [
            PY,
            "scripts/validate_standalone_physical_insertion.py",
            "--shapes",
            "all",
            "--dx-mm",
            2,
            "--out",
            out / "standalone_misaligned.json",
        ],
        a.dry_run,
    )
    run(
        [
            PY,
            "scripts/evaluate_panda_template_pose.py",
            "--dataset",
            "data/panda_native_topdown_contrast_test",
            "--camera",
            "topdown",
            "--segmentation",
            "models/segmentation_panda_native_topdown_contrast.pt",
            "--out",
            out / "panda_native_template_predmask.json",
        ],
        a.dry_run,
    )
    if not a.skip_panda:
        panda_episodes = int(a.panda_episodes) if a.panda_episodes is not None else (2 if a.quick else 40)
        if panda_episodes < 1:
            raise ValueError("--panda-episodes must be positive")
        for mask_source in ("ground_truth", "predicted"):
            for method in ("oracle", "sfss", "sfms", "mfms"):
                command = [
                    PY,
                    "scripts/panda_evaluate_controller.py",
                    "--method",
                    method,
                    "--task",
                    "insertion",
                    "--native-camera",
                    "--native-template-vsn",
                    "--camera-ignore-robot-occlusion",
                    "--mask_source",
                    mask_source,
                    "--split",
                    "test_unseen",
                    "--episodes",
                    panda_episodes,
                    "--execution-mode",
                    "dynamic",
                    "--seed",
                    4100,
                    "--out",
                    out / f"panda_matrix_v2_{method}_{mask_source}_insertion",
                ]
                if method == "sfms":
                    command.extend(["--policy", "models/sfms_mesh_v2_rl_best_compatible.pt"])
                elif method == "mfms":
                    command.extend(["--policy", "models/mfms_mesh_v2_teacher_compatible.pt"])
                if mask_source == "predicted" and method != "oracle":
                    command.extend(
                        [
                            "--segmentation",
                            "models/segmentation_panda_native_topdown_contrast.pt",
                        ]
                    )
                run(command, a.dry_run)
        run(
            [
                PY,
                "scripts/summarize_panda_matrix.py",
                "--root",
                out,
                "--out",
                out / "panda_dynamic_insertion_matrix",
            ],
            a.dry_run,
        )
    run([PY, "scripts/summarize_benchmarks.py", "--help"], a.dry_run)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "quick": bool(a.quick),
        "dry_run": bool(a.dry_run),
        "skip_panda": bool(a.skip_panda),
        "started_unix": started,
        "completed_unix": time.time(),
        "runtime": run_metadata(3100),
        "commands": RUN_COMMANDS,
        "inputs": {
            path: file_sha256(path)
            for path in (
                "configs/mesh_insertion_tight.yaml",
                "models/segmentation_mesh_v2_ce.pt",
                "models/position_mesh_v2_geometric.pt",
                "models/orientation_mesh_v2_symmetry_hybrid.pt",
                "models/sfms_mesh_v2_rl_best_compatible.pt",
                "models/mfms_mesh_v2_teacher_compatible.pt",
                "models/segmentation_panda_native_topdown_contrast.pt",
                "data/panda_native_topdown_contrast_test/manifest.json",
            )
        },
    }
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Final benchmark workflow completed." if not a.dry_run else "Dry-run command graph completed.")


if __name__ == "__main__":
    main()
