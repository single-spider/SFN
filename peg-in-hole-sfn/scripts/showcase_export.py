#!/usr/bin/env python
"""Export a deterministic, provenance-carrying manifest for the public showcase.

The public file combines the historical Cartesian/mesh results with the
corrected dynamic Panda matrix passed through ``--panda-root``.  No earlier
Panda matrix is exported, avoiding mixed simulator configurations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path("artifacts/software_completion_20260713")
REPORT = RELEASE / "FINAL_SOFTWARE_REPORT.md"
BENCHMARK = RELEASE / "final_benchmark_release" / "benchmark_manifest.json"
MESH_FILES = {
    "clean": RELEASE / "final_benchmark_release" / "mesh_clean" / "summary.json",
    "severity_4": RELEASE / "final_benchmark_release" / "mesh_severity4" / "summary.json",
}
MESH_PER_SHAPE_FILES = {
    "clean": RELEASE / "final_benchmark_release" / "mesh_clean" / "per_shape.csv",
    "severity_4": RELEASE / "final_benchmark_release" / "mesh_severity4" / "per_shape.csv",
}
SPLIT_MANIFESTS = {
    "train_seen": Path("data/mesh_v2_train_seen_clean_release/manifest.json"),
    "validation_unseen": Path("data/mesh_v2_validation_unseen_clean_release/manifest.json"),
    "test_unseen": Path("data/mesh_v2_test_unseen_clean_release/manifest.json"),
}


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file without relying on repository state."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_record(path: Path) -> dict[str, str]:
    relative = path.relative_to(ROOT).as_posix()
    return {"path": relative, "sha256": sha256(path)}


def compact_method_metrics(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep public metrics small while retaining trial counts and uncertainty."""
    result = {}
    for method, values in sorted(summary["methods"].items()):
        interval = values["success_rate_wilson_95"]
        result[method] = {
            "episodes": values["episodes"],
            "successes": values["successes"],
            "success_rate": values["success_rate"],
            "success_rate_wilson_95": [interval["low"], interval["high"]],
            "mean_final_xy_error_mm": values["mean_final_xy_error_mm"],
            "mean_final_yaw_error_deg": values["mean_final_yaw_error_deg"],
            "mean_steps": values["mean_steps"],
        }
    return result


def compact_per_shape_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Expose the small per-shape CSV without copying episode-level traces."""
    return [
        {
            "shape": row["shape"],
            "method": row["method"],
            "episodes": int(row["episodes"]),
            "successes": int(row["successes"]),
            "success_rate": float(row["success_rate"]),
            "mean_final_xy_error_mm": float(row["mean_final_xy_error_mm"]),
            "mean_final_yaw_error_deg": float(row["mean_final_yaw_error_deg"]),
        }
        for row in sorted(rows, key=lambda row: (row["shape"], row["method"]))
    ]


def extract_cartesian_report_metrics(report: str) -> list[dict[str, Any]]:
    """Extract only explicitly quantified Cartesian claims from the final report.

    The report is authoritative for this historical backend; values are never
    inferred from prose or substituted with mesh measurements.
    """
    patterns = (
        (
            "strict_xy_a2c",
            r"Strict X-Y A2C in the earlier Cartesian setup \| (\d+)/(\d+); mean X-Y ([0-9.]+) mm",
        ),
    )
    metrics = []
    for identifier, pattern in patterns:
        match = re.search(pattern, report)
        if not match:
            raise ValueError(f"Final report no longer contains expected {identifier} metric")
        successes, episodes, error = match.groups()
        metrics.append(
            {
                "id": identifier,
                "status": "historical_cartesian",
                "successes": int(successes),
                "episodes": int(episodes),
                "mean_final_xy_error_mm": float(error),
            }
        )
    return metrics


def shape_splits() -> list[dict[str, str]]:
    shapes: list[dict[str, str]] = []
    seen: set[str] = set()
    for expected_split, relative_path in SPLIT_MANIFESTS.items():
        manifest = load_json(ROOT / relative_path)
        if manifest["split"] != expected_split:
            raise ValueError(f"Split mismatch in {relative_path}: {manifest['split']}")
        for name in manifest["shapes"]:
            if name in seen:
                raise ValueError(f"Shape {name!r} is assigned to more than one split")
            seen.add(name)
            shapes.append({"name": name, "split": expected_split})
    if len(shapes) != 16:
        raise ValueError(f"Expected 16 shape names across release splits, found {len(shapes)}")
    return shapes


def build_manifest(panda_root: Path) -> dict[str, Any]:
    report_path = ROOT / REPORT
    benchmark_path = ROOT / BENCHMARK
    benchmark = load_json(benchmark_path)
    report = report_path.read_text(encoding="utf-8")
    mesh_summaries = {name: load_json(ROOT / path) for name, path in MESH_FILES.items()}
    mesh_per_shape = {name: load_csv(ROOT / path) for name, path in MESH_PER_SHAPE_FILES.items()}
    panda_matrix = panda_root / "panda_dynamic_insertion_matrix" / "summary.json"
    panda_matrix_csv = panda_root / "panda_dynamic_insertion_matrix" / "matrix.csv"
    panda = load_json(panda_matrix)
    panda_csv = load_csv(panda_matrix_csv)

    json_panda_keys = {(row["method"], row["mask_source"]) for row in panda["matrix"]}
    csv_panda_keys = {(row["method"], row["mask_source"]) for row in panda_csv}
    if json_panda_keys != csv_panda_keys:
        raise ValueError("Panda JSON and CSV matrix cells disagree")

    sources = [
        report_path,
        benchmark_path,
        *(ROOT / path for path in MESH_FILES.values()),
        *(ROOT / path for path in MESH_PER_SHAPE_FILES.values()),
        panda_matrix,
        panda_matrix_csv,
    ]
    sources.extend(ROOT / path for path in SPLIT_MANIFESTS.values())
    manifest = {
        "schema_version": "showcase-public/v2",
        "source_release": "software_completion_20260713",
        "benchmark_status": "corrected_panda_and_mesh",
        "sources": [source_record(path) for path in sources],
        "source_revision": benchmark["runtime"]["git"]["commit"],
        "source_revision_dirty": benchmark["runtime"]["git"]["dirty"],
        "shape_splits": shape_splits(),
        "metrics": {
            "cartesian": extract_cartesian_report_metrics(report),
            "mesh": {
                profile: {
                    "backend": "mesh_faithful_synthetic",
                    "mask_source": "predicted",
                    "episode_budget_per_method": summary["episode_budget_per_method"],
                    "methods": compact_method_metrics(summary),
                    "per_shape": compact_per_shape_metrics(mesh_per_shape[profile]),
                }
                for profile, summary in mesh_summaries.items()
            },
        },
        "panda": {
            "status": "corrected_current",
            "backend": panda["backend"],
            "test_shapes": panda["test_shapes"],
            "source_path": panda_matrix.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(panda_matrix),
            "source_csv_path": panda_matrix_csv.relative_to(ROOT).as_posix(),
            "source_csv_sha256": sha256(panda_matrix_csv),
            "results": [
                {
                    key: row[key]
                    for key in (
                        "method",
                        "mask_source",
                        "episodes",
                        "successes",
                        "success_rate",
                        "success_rate_ci95_low",
                        "success_rate_ci95_high",
                        "mean_final_xy_error_mm",
                        "mean_final_yaw_error_deg",
                    )
                }
                for row in panda["matrix"]
            ],
        },
    }
    return manifest


def verify_manifest(manifest: dict[str, Any]) -> None:
    assert manifest["benchmark_status"] == "corrected_panda_and_mesh"
    assert len(manifest["shape_splits"]) == 16
    assert {item["split"] for item in manifest["shape_splits"]} == set(SPLIT_MANIFESTS)
    assert manifest["panda"]["status"] == "corrected_current"
    assert len(manifest["panda"]["results"]) == 8
    assert set(manifest["metrics"]["mesh"]) == set(MESH_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/showcase/manifest.json")
    parser.add_argument(
        "--panda-root",
        type=Path,
        required=True,
        help="corrected run directory containing panda_dynamic_insertion_matrix",
    )
    parser.add_argument("--check", action="store_true", help="fail if --out is not the deterministic export")
    parser.add_argument("--self-test", action="store_true", help="validate the generated schema without writing")
    args = parser.parse_args()
    panda_root = args.panda_root if args.panda_root.is_absolute() else ROOT / args.panda_root
    manifest = build_manifest(panda_root)
    verify_manifest(manifest)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.self_test:
        return
    output = args.out if args.out.is_absolute() else ROOT / args.out
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Showcase manifest is stale: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
