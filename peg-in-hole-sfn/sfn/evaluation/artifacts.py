"""Canonical CSV/JSON artifact writing for benchmark runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_records_csv(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    """Write heterogeneous records with a stable union of columns."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for record in records for key in record})
    with destination.open("w", newline="", encoding="utf-8") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({str(key): _cell(value) for key, value in record.items()} for record in records)
    return destination


def write_evaluation_artifacts(
    output_dir: str | Path,
    episodes: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]] = (),
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ``episodes.csv``, ``steps.csv``, and optional ``summary.json``."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "episodes": write_records_csv(output / "episodes.csv", episodes),
        "steps": write_records_csv(output / "steps.csv", steps),
    }
    if summary is not None:
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["summary"] = summary_path
    return paths


class EvaluationArtifactWriter:
    """Incremental in-memory collector for episode and step artifacts."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.episodes: list[Mapping[str, Any]] = []
        self.steps: list[Mapping[str, Any]] = []

    def add_episode(self, record: Mapping[str, Any]) -> None:
        self.episodes.append(dict(record))

    def add_step(self, record: Mapping[str, Any]) -> None:
        self.steps.append(dict(record))

    def write(self, *, summary: Mapping[str, Any] | None = None) -> dict[str, Path]:
        return write_evaluation_artifacts(self.output_dir, self.episodes, self.steps, summary=summary)
