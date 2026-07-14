"""Read and summarize benchmark episode artifacts."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .statistics import paired_binary_counts, paired_bootstrap_difference, percentile_summary, wilson_interval

TRUE_VALUES = {"1", "true", "yes", "y"}


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def read_episode_csv(path: str | Path, *, method: str | None = None) -> list[dict[str, Any]]:
    """Read an episodes CSV, attaching a method label when it is absent."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    label = method or source.parent.name or source.stem
    for row in rows:
        row.setdefault("method", label)
    return rows


def summarize_method(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty benchmark")
    successes = [_boolean(record["success"]) for record in records]
    success_count = sum(successes)
    low, high = wilson_interval(success_count, len(records))
    result: dict[str, Any] = {
        "episodes": len(records),
        "successes": success_count,
        "success_rate": success_count / len(records),
        "success_ci_low": low,
        "success_ci_high": high,
    }
    for metric in ("steps", "reward", "final_xy_error_mm", "final_yaw_error_deg", "insertion_depth_mm"):
        values = [float(record[metric]) for record in records if record.get(metric) not in (None, "")]
        if values:
            result[metric] = percentile_summary(values)
    return result


def summarize_benchmarks(
    records: Iterable[dict[str, Any]],
    *,
    pair_keys: Sequence[str] = ("shape", "episode", "task"),
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize methods and, for two methods, their paired outcomes/metrics."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("method", "unknown"))].append(record)
    if not grouped:
        raise ValueError("no episode records supplied")
    report: dict[str, Any] = {"methods": {name: summarize_method(rows) for name, rows in sorted(grouped.items())}}
    if len(grouped) == 2:
        names = sorted(grouped)
        indexes = [{tuple(row.get(key, "") for key in pair_keys): row for row in grouped[name]} for name in names]
        common = sorted(set(indexes[0]) & set(indexes[1]))
        if common:
            left, right = ([indexes[i][key] for key in common] for i in range(2))
            comparison: dict[str, Any] = {
                "a": names[0],
                "b": names[1],
                "paired_episodes": len(common),
                "success": paired_binary_counts(
                    [_boolean(row["success"]) for row in left], [_boolean(row["success"]) for row in right]
                ),
            }
            for metric in ("steps", "reward", "final_xy_error_mm", "final_yaw_error_deg", "insertion_depth_mm"):
                pairs = [(a.get(metric), b.get(metric)) for a, b in zip(left, right, strict=True)]
                complete = [(float(a), float(b)) for a, b in pairs if a not in (None, "") and b not in (None, "")]
                if complete:
                    comparison[metric] = paired_bootstrap_difference(
                        [x for x, _ in complete], [y for _, y in complete], resamples=resamples, seed=seed
                    )
            report["comparison"] = comparison
    return report
