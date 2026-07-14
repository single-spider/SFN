#!/usr/bin/env python
"""Run the compact, paired release-ablation matrix and consolidate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import config_from_dict


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        # Reuse the project's dependency-free YAML fallback for ordinary
        # dataclass configs. The suite matrix itself is JSON because it needs
        # a list of nested run mappings.
        from sfn.config import _load_yaml_file

        value = _load_yaml_file(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return repr(value) if isinstance(value, str) else str(value)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    lines: list[str] = []

    def emit(mapping: dict[str, Any], indent: int) -> None:
        prefix = " " * indent
        for key, item in mapping.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                emit(item, indent + 2)
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")

    emit(value, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _run_summary(run: dict[str, Any], records: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    success = [str(row.get("success", "")).lower() == "true" for row in records]
    xy = [value for row in records if (value := _number(row.get("final_xy_error_mm"))) is not None]
    yaw = [value for row in records if (value := _number(row.get("final_yaw_error_deg"))) is not None]
    steps = [value for row in records if (value := _number(row.get("steps"))) is not None]
    return {
        "run_id": run["id"],
        "comparison": run["comparison"],
        "label": run["label"],
        "backend": run.get("backend", "cartesian"),
        "method": run.get("method", "sfss"),
        "task": run.get("task", "alignment"),
        "status": "completed",
        "episodes": len(records),
        "success_rate": _mean([float(item) for item in success]),
        "mean_final_xy_error_mm": _mean(xy),
        "mean_final_yaw_error_deg": _mean(yaw),
        "mean_steps": _mean(steps),
        "elapsed_s": elapsed_s,
    }


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("shape", "")), str(row.get("episode_seed", ""))


def _comparison_summary(name: str, members: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    run_rows = {member["id"]: [row for row in records if row["ablation_run_id"] == member["id"]] for member in members}
    seed_sets = [{_pair_key(row) for row in rows} for rows in run_rows.values()]
    common = set.intersection(*seed_sets) if seed_sets else set()
    metrics: dict[str, Any] = {}
    for member in members:
        rows = run_rows[member["id"]]
        by_key = {_pair_key(row): row for row in rows}
        metrics[member["label"]] = {
            "run_id": member["id"],
            "episode_count": len(rows),
            "paired_episode_count": len(common),
            "success_rate": _mean(
                [float(str(by_key[key].get("success", "")).lower() == "true") for key in sorted(common)]
            ),
            "mean_final_xy_error_mm": _mean(
                [value for key in sorted(common) if (value := _number(by_key[key].get("final_xy_error_mm"))) is not None]
            ),
            "mean_final_yaw_error_deg": _mean(
                [value for key in sorted(common) if (value := _number(by_key[key].get("final_yaw_error_deg"))) is not None]
            ),
        }
    return {
        "comparison": name,
        "paired": bool(common) and all(seed_set == seed_sets[0] for seed_set in seed_sets),
        "common_episode_count": len(common),
        "conditions": metrics,
    }


def _command(run: dict[str, Any], suite: dict[str, Any], resolved_config: Path, run_out: Path) -> list[str]:
    shapes = run.get("shapes", suite["shapes"])
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        str(resolved_config),
        "--backend",
        run.get("backend", "cartesian"),
        "--method",
        run.get("method", "sfss"),
        "--task",
        run.get("task", "alignment"),
        "--episodes",
        str(run.get("episodes", suite["episodes"])),
        "--shapes",
        ",".join(shapes),
        "--seed",
        str(suite["seed"]),
        "--out",
        str(run_out),
    ]
    if run.get("method", "sfss") == "sfss":
        command.extend(
            [
                "--mask_source",
                run.get("mask_source", "ground_truth"),
                "--position",
                suite["checkpoints"]["position"],
                "--orientation",
                suite["checkpoints"]["orientation"],
                "--confidence-mode",
                run.get("confidence_mode", "ignore"),
                "--sfss-gain-yaw",
                str(run.get("sfss_gain_yaw", 0.7)),
                "--ensemble-samples",
                str(run.get("ensemble_samples", 1)),
            ]
        )
        if run.get("robustness_profile") is not None:
            command.extend(["--robustness-profile", run["robustness_profile"]])
        if run.get("temporal_alpha") is not None:
            command.extend(["--temporal-alpha", str(run["temporal_alpha"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "release_ablations.json"))
    parser.add_argument("--out", default=None, help="Override suite.output")
    parser.add_argument("--skip-panda", action="store_true", help="Skip optional Panda smoke comparisons")
    parser.add_argument("--only", default=None, help="Comma-separated run IDs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without evaluating")
    args = parser.parse_args()

    source_path = Path(args.config).resolve()
    spec = _load_mapping(source_path)
    suite = spec["suite"]
    runs = list(spec["runs"])
    selected = {item.strip() for item in args.only.split(",")} if args.only else None
    runs = [run for run in runs if selected is None or run["id"] in selected]
    if selected and selected - {run["id"] for run in runs}:
        raise SystemExit(f"Unknown --only run(s): {', '.join(sorted(selected - {run['id'] for run in runs}))}")
    if args.skip_panda:
        runs = [run for run in runs if not str(run.get("backend", "cartesian")).startswith("panda_")]

    out = Path(args.out or suite["output"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    base_path = ROOT / suite["base_config"]
    base_config = _load_mapping(base_path)

    ids = [run["id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Ablation run IDs must be unique")
    manifest = {
        "source_config": str(source_path),
        "base_config": str(base_path.resolve()),
        "seed": suite["seed"],
        "run_ids": ids,
        "commands": {},
    }
    all_records: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run in runs:
        resolved = _merge(base_config, run.get("config_overrides", {}))
        resolved.setdefault("project", {})["seed"] = suite["seed"]
        config_from_dict(resolved)  # strict validation before spawning work
        resolved_path = out / "config" / f"{run['id']}.yaml"
        _write_yaml(resolved_path, resolved)
        run_out = out / "runs" / run["id"]
        command = _command(run, suite, resolved_path, run_out)
        manifest["commands"][run["id"]] = command
        print(" ".join(command))
        if args.dry_run:
            continue
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        elapsed = time.perf_counter() - started
        log = f"COMMAND: {' '.join(command)}\nEXIT_CODE: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        log_path = out / "logs" / f"{run['id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log, encoding="utf-8")
        if completed.returncode != 0:
            failure = {
                "run_id": run["id"],
                "comparison": run["comparison"],
                "label": run["label"],
                "status": "failed_optional" if run.get("optional") else "failed",
                "returncode": completed.returncode,
                "elapsed_s": elapsed,
                "log": str(log_path),
            }
            run_summaries.append(failure)
            if run.get("optional"):
                continue
            raise SystemExit(f"Required run {run['id']} failed; see {log_path}")
        records = _read_csv(run_out / "episodes.csv")
        steps = _read_csv(run_out / "steps.csv")
        tags = {
            "ablation_run_id": run["id"],
            "ablation_comparison": run["comparison"],
            "ablation_label": run["label"],
        }
        for row in records:
            row.update(tags)
        for row in steps:
            row.update(tags)
        all_records.extend(records)
        all_steps.extend(steps)
        run_summaries.append(_run_summary(run, records, elapsed))

    (out / "suite_config.json").write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"Validated {len(runs)} runs; dry run wrote configs and manifest to {out}")
        return

    _write_union_csv(out / "raw_records.csv", all_records)
    _write_union_csv(out / "raw_steps.csv", all_steps)
    _write_union_csv(out / "summary.csv", run_summaries)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completed_ids = {row["ablation_run_id"] for row in all_records}
    for run in runs:
        if run["id"] in completed_ids:
            grouped[run["comparison"]].append(run)
    comparisons = [_comparison_summary(name, members, all_records) for name, members in grouped.items()]
    summary = {
        "suite": "compact_release_ablations",
        "seed": suite["seed"],
        "requested_runs": len(runs),
        "completed_runs": sum(item["status"] == "completed" for item in run_summaries),
        "failed_optional_runs": [item["run_id"] for item in run_summaries if item["status"] == "failed_optional"],
        "raw_episode_records": len(all_records),
        "raw_step_records": len(all_steps),
        "runs": run_summaries,
        "comparisons": comparisons,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
