"""Panda-specific persisted artifact helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.artifacts import write_records_csv


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def panda_per_shape_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Panda episodes by shape and method without hiding failures."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("shape", "unknown")), str(record.get("method", "unknown")))].append(record)

    output = []
    for (shape, method), rows in sorted(grouped.items()):
        successes = sum(bool(row.get("success", False)) for row in rows)
        failure_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if not bool(row.get("success", False)):
                failure_counts[str(row.get("failure_state") or row.get("failure_category") or "unknown")] += 1
        output.append(
            {
                "shape": shape,
                "method": method,
                "episodes": len(rows),
                "successes": successes,
                "failures": len(rows) - successes,
                "success_rate": successes / len(rows),
                "failure_states": dict(sorted(failure_counts.items())),
                "mean_steps": _mean(rows, "steps"),
                "mean_final_xy_error_mm": _mean(rows, "final_xy_error_mm"),
                "mean_final_yaw_error_deg": _mean(rows, "final_yaw_error_deg"),
                "mean_tracking_error_mm": _mean(rows, "tracking_error_mm"),
                "mean_insertion_depth_mm": _mean(rows, "insertion_depth_mm"),
                "mean_lateral_drift_mm": _mean(rows, "lateral_drift_mm"),
                "mean_contact_count": _mean(rows, "contact_count"),
                "max_contact_force": max(
                    (float(row["max_contact_force"]) for row in rows if row.get("max_contact_force") is not None),
                    default=None,
                ),
                "max_penetration_mm": max(
                    (float(row["max_penetration_mm"]) for row in rows if row.get("max_penetration_mm") is not None),
                    default=None,
                ),
                "min_joint_limit_margin": min(
                    (
                        float(row["min_joint_limit_margin"])
                        for row in rows
                        if row.get("min_joint_limit_margin") is not None
                    ),
                    default=None,
                ),
            }
        )
    return output


def write_panda_per_shape(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    return write_records_csv(path, panda_per_shape_rows(records))
