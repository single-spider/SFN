import csv
import json

import pytest
from sfn.evaluation.artifacts import EvaluationArtifactWriter
from sfn.evaluation.reporting import read_episode_csv, summarize_benchmarks
from sfn.evaluation.statistics import (
    paired_binary_counts,
    paired_bootstrap_difference,
    percentile_summary,
    wilson_interval,
)


def test_statistics_are_deterministic_and_paired():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)
    assert percentile_summary([0, 10, 20], percentiles=(25, 50, 75)) == {
        "count": 3,
        "mean": 10.0,
        "std": pytest.approx(8.16496580927726),
        "min": 0.0,
        "max": 20.0,
        "p25": 5.0,
        "p50": 10.0,
        "p75": 15.0,
    }
    boot = paired_bootstrap_difference([3, 4, 5], [1, 2, 3], resamples=100, seed=7)
    assert boot["difference"] == 2.0
    assert boot["ci_low"] == boot["ci_high"] == 2.0
    counts = paired_binary_counts([True, True, False, False], [True, False, True, False])
    assert (counts["both_success"], counts["a_only"], counts["b_only"], counts["neither_success"]) == (1, 1, 1, 1)
    assert counts["mcnemar_exact_p"] == 1.0


def test_artifact_writer_and_summary(tmp_path):
    writer = EvaluationArtifactWriter(tmp_path / "a")
    writer.add_episode({"shape": "round", "episode": 0, "task": "alignment", "success": True, "steps": 2})
    writer.add_step({"episode": 0, "action": [1, 2]})
    paths = writer.write(summary={"ok": True})
    assert set(paths) == {"episodes", "steps", "summary"}
    assert json.loads(paths["summary"].read_text()) == {"ok": True}
    with paths["steps"].open(newline="", encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["action"] == "[1,2]"

    b = tmp_path / "b" / "episodes.csv"
    b.parent.mkdir()
    b.write_text("shape,episode,task,success,steps\nround,0,alignment,false,4\n", encoding="utf-8")
    rows = read_episode_csv(paths["episodes"], method="A") + read_episode_csv(b, method="B")
    report = summarize_benchmarks(rows, resamples=100, seed=1)
    assert report["methods"]["A"]["success_rate"] == 1.0
    assert report["comparison"]["success"]["a_only"] == 1
    assert report["comparison"]["steps"]["difference"] == -2.0


def test_invalid_statistical_inputs():
    with pytest.raises(ValueError):
        wilson_interval(2, 1)
    with pytest.raises(ValueError):
        percentile_summary([])
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1], [1, 2])
    with pytest.raises(TypeError):
        paired_binary_counts([1], [True])
