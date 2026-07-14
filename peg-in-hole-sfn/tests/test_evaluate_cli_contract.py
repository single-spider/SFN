import json
import sys

import scripts.evaluate as evaluate_cli


def _episode(method: str) -> dict:
    return {
        "method": method,
        "shape": "synthetic-square",
        "success": True,
        "steps": 1,
        "final_xy_error_mm": 0.0,
        "final_yaw_error_deg": 0.0,
    }


def test_method_all_runs_all_five_variants_with_distinct_policies(monkeypatch, tmp_path):
    calls = []

    def oracle(**kwargs):
        calls.append(("oracle", None, kwargs["episodes"]))
        return [_episode("oracle")], [{"method": "oracle"}]

    def sfss(**kwargs):
        method = "sfss_recursive" if kwargs["recursive"] else "sfss_one_step"
        calls.append((method, None, kwargs["episodes"]))
        return [_episode(method)], [{"method": method}]

    def sfms(**kwargs):
        calls.append(("sfms", str(kwargs["policy_path"]), kwargs["episodes"]))
        return [_episode("sfms")], [{"method": "sfms"}]

    def mfms(**kwargs):
        calls.append(("mfms", str(kwargs["policy_path"]), kwargs["episodes"]))
        return [_episode("mfms")], [{"method": "mfms"}]

    monkeypatch.setattr(evaluate_cli, "evaluate_oracle", oracle)
    monkeypatch.setattr(evaluate_cli, "evaluate_sfss", sfss)
    monkeypatch.setattr(evaluate_cli, "evaluate_sfms", sfms)
    monkeypatch.setattr(evaluate_cli, "evaluate_mfms", mfms)
    monkeypatch.setattr(evaluate_cli, "_build_vsn", lambda args, seed: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--method",
            "all",
            "--episodes",
            "2",
            "--sfms-policy",
            "sfms.pt",
            "--mfms-policy",
            "mfms.pt",
            "--shapes",
            "synthetic-square",
            "--out",
            str(tmp_path),
        ],
    )

    evaluate_cli.main()

    assert calls == [
        ("oracle", None, 2),
        ("sfss_one_step", None, 2),
        ("sfss_recursive", None, 2),
        ("sfms", "sfms.pt", 2),
        ("mfms", "mfms.pt", 2),
    ]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["methods"]) == {"oracle", "sfss_one_step", "sfss_recursive", "sfms", "mfms"}
    assert summary["episode_budget_per_method"] == 2
    assert (tmp_path / "steps.csv").is_file()
