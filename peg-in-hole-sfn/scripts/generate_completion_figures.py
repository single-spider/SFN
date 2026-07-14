#!/usr/bin/env python
"""Generate final robustness, history, and Panda result plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "software_completion_20260713"
OUT = ART / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def severity_figure() -> list[dict]:
    methods = ["sfss_recursive", "sfms", "mfms"]
    rows = []
    for level in range(5):
        summary = load(ART / f"severity_{level}_all_methods_insertion_40" / "summary.json")
        for method in methods:
            rows.append({"severity": level, "method": method, **summary["methods"][method]})
    fields = [
        "severity",
        "method",
        "episodes",
        "successes",
        "success_rate",
        "success_rate_ci95_low",
        "success_rate_ci95_high",
        "mean_steps",
        "mean_final_xy_error_mm",
        "mean_final_yaw_error_deg",
    ]
    write_rows(OUT / "severity_curve.csv", fields, rows)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        x = [row["severity"] for row in selected]
        y = [100 * row["success_rate"] for row in selected]
        low = [max(0.0, 100 * (row["success_rate"] - row["success_rate_ci95_low"])) for row in selected]
        high = [max(0.0, 100 * (row["success_rate_ci95_high"] - row["success_rate"])) for row in selected]
        axis.errorbar(
            x,
            y,
            yerr=[low, high],
            marker="o",
            capsize=3,
            label=method.replace("_", " ").upper(),
        )
    axis.set(
        xlabel="Combined disturbance severity",
        ylabel="Insertion success (%)",
        xticks=range(5),
        ylim=(75, 102),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "severity_success.png", dpi=180)
    plt.close(figure)
    return rows


def history_figure() -> list[dict]:
    rows = []
    for history_len in [1, 2, 4, 8]:
        summary = load(ART / f"mfms_history_paired_h{history_len}_burst5_insertion_80" / "summary.json")
        rows.append({"history_len": history_len, **summary["methods"]["mfms"]})
    fields = [
        "history_len",
        "episodes",
        "successes",
        "success_rate",
        "success_rate_ci95_low",
        "success_rate_ci95_high",
        "mean_steps",
        "mean_final_xy_error_mm",
        "mean_final_yaw_error_deg",
    ]
    write_rows(OUT / "mfms_history_ablation.csv", fields, rows)

    figure, axis = plt.subplots(figsize=(6.2, 4))
    axis.plot(
        [row["history_len"] for row in rows],
        [100 * row["success_rate"] for row in rows],
        marker="o",
    )
    axis.set(
        xlabel="MFMS history length",
        ylabel="Burst-occlusion insertion success (%)",
        xticks=[1, 2, 4, 8],
        ylim=(45, 102),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT / "mfms_history_ablation.png", dpi=180)
    plt.close(figure)
    return rows


def panda_figure() -> list[dict]:
    rows = load(ART / "panda_dynamic_insertion_matrix" / "summary.json")["matrix"]
    methods = ["oracle", "sfss", "sfms", "mfms"]
    masks = ["ground_truth", "predicted"]
    x = list(range(len(methods)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for index, mask in enumerate(masks):
        selected = [
            next(row for row in rows if row["method"] == method and row["mask_source"] == mask) for method in methods
        ]
        y = [100 * row["success_rate"] for row in selected]
        low = [max(0.0, 100 * (row["success_rate"] - row["success_rate_ci95_low"])) for row in selected]
        high = [max(0.0, 100 * (row["success_rate_ci95_high"] - row["success_rate"])) for row in selected]
        axis.bar(
            [position + (index - 0.5) * width for position in x],
            y,
            width,
            yerr=[low, high],
            capsize=3,
            label=mask.replace("_", " ").title(),
        )
    axis.set(
        ylabel="Dynamic insertion success (%)",
        xticks=x,
        xticklabels=[method.upper() for method in methods],
        ylim=(65, 105),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "panda_dynamic_insertion_matrix.png", dpi=180)
    plt.close(figure)
    return rows


def main() -> None:
    severity = severity_figure()
    history = history_figure()
    panda = panda_figure()
    print(
        json.dumps(
            {
                "out": str(OUT),
                "severity_rows": len(severity),
                "history_rows": len(history),
                "panda_rows": len(panda),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
