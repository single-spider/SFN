#!/usr/bin/env python
"""Launch and monitor position + orientation training in bounded batches.

This is intentionally small and dependency-free so it can run for hours from a
terminal/background process without Codex/API timeouts.  It starts both
trainers, records stdout/stderr to files, polls their JSONL metrics, writes a
machine-readable monitor status, and terminates a run if the first completed
epoch is clearly non-ideal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"parse_error": last[-500:]}


def _backup_outputs(paths: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.exists():
            shutil.copy2(p, backup_dir / p.name)


def _job_command(task: str, args: argparse.Namespace) -> tuple[list[str], Path, Path]:
    out = ROOT / "models" / f"{task}.pt"
    metrics = out.with_name(out.stem + ".metrics.jsonl")
    script = ROOT / "scripts" / f"train_{task}.py"
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--dataset",
        args.dataset,
        "--val-dataset",
        args.val_dataset,
        "--out",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--base-channels",
        str(args.base_channels),
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--val-limit",
        str(args.val_limit),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.position_seed if task == "position" else args.orientation_seed),
        "--no-progress",
    ]
    if args.amp:
        cmd.append("--amp")
    return cmd, out, metrics


def _summarize_metric(task: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"task": task, "status": "waiting_for_first_epoch"}
    val = row.get("val") or {}
    if task == "position":
        return {
            "task": task,
            "epoch": row.get("epoch"),
            "best_epoch": row.get("best_epoch"),
            "mean_radial_error_mm": val.get("mean_radial_error_mm"),
            "exact_cell_accuracy": val.get("exact_cell_accuracy"),
            "within_1_cell_accuracy": val.get("within_1_cell_accuracy"),
            "best_metric_value": row.get("best_metric_value"),
        }
    return {
        "task": task,
        "epoch": row.get("epoch"),
        "best_epoch": row.get("best_epoch"),
        "mean_abs_error_deg": val.get("mean_abs_error_deg"),
        "within_2_deg_accuracy": val.get("within_2_deg_accuracy"),
        "within_4_deg_accuracy": val.get("within_4_deg_accuracy"),
        "best_metric_value": row.get("best_metric_value"),
    }


def _nonideal(task: str, row: dict[str, Any] | None) -> str | None:
    if row is None or int(row.get("epoch") or 0) < 1:
        return None
    val = row.get("val") or {}
    if task == "position":
        err = float(val.get("mean_radial_error_mm", 999.0))
        if err > 0.5:
            return f"position radial error too high after epoch {row.get('epoch')}: {err:.4f} mm"
    else:
        err = float(val.get("mean_abs_error_deg", 999.0))
        within2 = float(val.get("within_2_deg_accuracy", 0.0))
        if err > 1.5 or within2 < 0.99:
            return f"orientation metric non-ideal after epoch {row.get('epoch')}: mean_abs={err:.4f} deg within2={within2:.4f}"
    return None


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    deadline = time.time() + 20
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Run monitored position + orientation training batch")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--val-limit", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--dataset", default="data/train_seen_40k_edge_fast")
    ap.add_argument("--val-dataset", default="data/val_unseen_4k_edge_orientable")
    ap.add_argument("--position-seed", type=int, default=102)
    ap.add_argument("--orientation-seed", type=int, default=103)
    ap.add_argument("--poll-sec", type=int, default=300)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--fresh", action="store_true", help="Back up existing outputs before starting a non-resume run.")
    args = ap.parse_args()

    run_id = datetime.now().strftime("po_%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "artifacts" / "training_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    jobs: dict[str, dict[str, Any]] = {}
    output_paths: list[Path] = []
    for task in ["position", "orientation"]:
        cmd, out, metrics = _job_command(task, args)
        output_paths.extend(
            [out, out.with_name(out.stem + ".last" + out.suffix), metrics, out.with_name(out.stem + ".summary.json")]
        )
        jobs[task] = {"cmd": cmd, "out": out, "metrics": metrics}

    if args.fresh:
        _backup_outputs(output_paths, run_dir / "backup_existing_models")
        # Avoid a stale first monitor snapshot from an earlier run.  The
        # trainers also clear their metrics on non-resume starts, but doing it
        # here keeps monitor_status.json honest from second zero.
        for p in output_paths:
            if p.name.endswith((".metrics.jsonl", ".summary.json")) and p.exists():
                p.unlink()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for task, job in jobs.items():
        log = (run_dir / f"{task}.log").open("w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(job["cmd"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        job["pid"] = proc.pid
        job["proc"] = proc
        job["log_file"] = str(run_dir / f"{task}.log")
        job["log_handle"] = log

    history_path = run_dir / "monitor_history.jsonl"
    status_path = run_dir / "monitor_status.json"
    stop_reasons: dict[str, str] = {}
    started = time.time()

    while True:
        all_done = True
        snapshot: dict[str, Any] = {
            "time_unix": time.time(),
            "elapsed_sec": time.time() - started,
            "run_dir": str(run_dir),
            "jobs": {},
            "stop_reasons": stop_reasons,
        }
        for task, job in jobs.items():
            proc: subprocess.Popen = job["proc"]
            row = _read_last_jsonl(job["metrics"])
            reason = _nonideal(task, row)
            if reason and task not in stop_reasons and proc.poll() is None:
                stop_reasons[task] = reason
                _terminate(proc)
            rc = proc.poll()
            if rc is None:
                all_done = False
            snapshot["jobs"][task] = {
                "pid": job["pid"],
                "returncode": rc,
                "cmd": job["cmd"],
                "log_file": job["log_file"],
                "metrics_file": str(job["metrics"]),
                "checkpoint": str(job["out"]),
                "latest": _summarize_metric(task, row),
            }
        status_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, sort_keys=True) + "\n")
        if all_done:
            break
        time.sleep(max(5, int(args.poll_sec)))

    for job in jobs.values():
        try:
            job["log_handle"].close()
        except Exception:
            pass
    return 2 if stop_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
