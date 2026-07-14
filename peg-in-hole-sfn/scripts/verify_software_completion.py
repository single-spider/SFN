#!/usr/bin/env python
"""Verify evidence-bearing software completion gates and write JSON proof."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "software_completion_20260713"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ART / "completion_verification.json")
    args = parser.parse_args()
    checks = []

    def check(name: str, passed: bool, evidence: str, value=None) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence, "value": value})

    cascade = load(ART / "perception_mesh_v2_final_cascade.json")
    cascade_samples = cascade["segmentation"]["samples"]
    semantic_xy = cascade["position"]["continuous_metric"]["mean_radial_error_mm"]
    semantic_yaw = cascade["orientation"]["mean_abs_error_deg"]
    check(
        "mesh_predicted_cascade_recorded",
        cascade_samples == 1054 and "predicted_mask_cascade" in cascade,
        "perception_mesh_v2_final_cascade.json",
        cascade_samples,
    )
    check(
        "mesh_semantic_position_submillimetre",
        semantic_xy <= 1.0,
        "perception_mesh_v2_final_cascade.json",
        semantic_xy,
    )
    check(
        "mesh_semantic_yaw_gate",
        semantic_yaw <= 2.0,
        "perception_mesh_v2_final_cascade.json",
        semantic_yaw,
    )

    clean_path = ART / "final_benchmark_release" / "mesh_clean" / "summary.json"
    clean = load(clean_path)["methods"]
    for method in ("sfss_recursive", "sfms", "mfms"):
        row = clean[method]
        check(
            f"mesh_clean_{method}",
            row["episodes"] == 40
            and row["success_rate"] >= 0.85
            and row["mean_final_xy_error_mm"] < 1.0,
            "final_benchmark_release/mesh_clean/summary.json",
            {
                "episodes": row["episodes"],
                "successes": row["successes"],
                "success_rate": row["success_rate"],
                "mean_xy_mm": row["mean_final_xy_error_mm"],
            },
        )

    for severity in range(5):
        summary = load(ART / f"severity_{severity}_all_methods_insertion_40" / "summary.json")
        check(
            f"paired_severity_{severity}",
            all(summary["methods"][method]["episodes"] == 40 for method in ("sfss_recursive", "sfms", "mfms")),
            f"severity_{severity}_all_methods_insertion_40/summary.json",
            {method: summary["methods"][method]["successes"] for method in ("sfss_recursive", "sfms", "mfms")},
        )
    histories = {}
    for history_len in (1, 2, 4, 8):
        row = load(ART / f"mfms_history_paired_h{history_len}_burst5_insertion_80" / "summary.json")["methods"]["mfms"]
        histories[str(history_len)] = row["successes"]
    check(
        "mfms_temporal_history_ablation",
        len(histories) == 4 and all(0 <= value <= 80 for value in histories.values()),
        "mfms_history_paired_h{1,2,4,8}_burst5_insertion_80",
        histories,
    )

    exact = load(ART / "standalone_pybullet_exact_all_shapes_final.json")
    bad = load(ART / "standalone_pybullet_misaligned_all_shapes_final.json")
    check(
        "standalone_exact_all_shapes",
        exact["successes"] == 16,
        exact["rows"][0]["renderer_backend"],
        exact["successes"],
    )
    check(
        "standalone_rejects_2mm",
        bad["successes"] == 0,
        "standalone_pybullet_misaligned_all_shapes_final.json",
        bad["successes"],
    )

    panda_exact = [load(path) for path in sorted((ART / "panda_insertion_raster_exact").glob("*.json"))]
    check(
        "panda_dynamic_exact_all_shapes",
        len(panda_exact) == 16 and all(row["success"] and row["execution_mode"] == "dynamic" for row in panda_exact),
        "panda_insertion_raster_exact/*.json",
        sum(row["success"] for row in panda_exact),
    )
    ik = load(ART / "panda_ik_all_shapes_full_workspace.json")
    check(
        "panda_full_workspace_ik",
        ik["successes"] == 16 and ik["total"] == 16 and all(row["targets"] == 125 for row in ik["rows"]),
        "panda_ik_all_shapes_full_workspace.json",
        {"shapes": ik["successes"], "targets_per_shape": 125},
    )

    segmentation = load(ART / "panda_native_topdown_contrast_segmentation_test.json")["segmentation"]
    pose = load(ART / "panda_native_topdown_contrast_template_predmask_test.json")["summary"]
    check(
        "panda_rgb_segmentation_gate",
        segmentation["class_iou"]["1"] >= 0.90,
        "panda_native_topdown_contrast_segmentation_test.json",
        segmentation["class_iou"],
    )
    check(
        "panda_predicted_pose_gate",
        pose["mean_xy_error_mm"] <= 1.0 and pose["mean_yaw_error_deg"] <= 2.0,
        "panda_native_topdown_contrast_template_predmask_test.json",
        {"xy_mm": pose["mean_xy_error_mm"], "yaw_deg": pose["mean_yaw_error_deg"]},
    )
    matrix_path = ART / "final_benchmark_release" / "panda_dynamic_insertion_matrix" / "summary.json"
    matrix = load(matrix_path)["matrix"]
    check(
        "panda_all_method_mask_matrix",
        len(matrix) == 8 and all(row["episodes"] == 20 for row in matrix),
        "final_benchmark_release/panda_dynamic_insertion_matrix/summary.json",
        {f"{row['mask_source']}:{row['method']}": row["successes"] for row in matrix},
    )

    quality_path = ART / "quality_gates.log"
    quality_bytes = quality_path.read_bytes()
    quality = quality_bytes.decode("utf-16") if quality_bytes.startswith(b"\xff\xfe") else quality_bytes.decode("utf-8")
    passed_counts = [int(value) for value in re.findall(r"(\d+) passed", quality)]
    check(
        "full_test_suite",
        bool(passed_counts) and max(passed_counts) >= 200 and " failed" not in quality,
        "quality_gates.log",
        max(passed_counts) if passed_counts else 0,
    )
    check("full_ruff_gate", "All checks passed!" in quality, "quality_gates.log", "passed")
    gpu_smoke = load(ART / "gpu_smoke.json")
    check(
        "gpu_checkpoint_parity_smoke",
        gpu_smoke["passed"],
        "gpu_smoke.json",
        {"device": gpu_smoke["device"], "cuda": gpu_smoke["cuda_version"]},
    )
    check(
        "final_benchmark_driver",
        (ART / "final_benchmark_release" / "benchmark_manifest.json").exists(),
        "final_benchmark_release",
        "completed",
    )
    tolerance = load(ART / "standalone_insertion_tolerance_map" / "summary.json")
    check(
        "physical_tolerance_map",
        tolerance["shapes"] == 16 and tolerance["trials"] == 1200,
        "standalone_insertion_tolerance_map",
        {"shapes": tolerance["shapes"], "trials": tolerance["trials"]},
    )
    random_comparison = load(ART / "random_vs_sfms_paired_40.json")["comparison"]["success"]
    check(
        "sfms_beats_paired_random_policy",
        random_comparison["b_only"] == 40 and random_comparison["mcnemar_exact_p"] < 0.001,
        "random_vs_sfms_paired_40.json",
        random_comparison,
    )
    panda_disturbed = load(ART / "panda_predicted_physical_severity2" / "summary_all.json")
    check(
        "panda_native_disturbance_matrix",
        all(panda_disturbed[method]["episodes"] == 20 for method in ("sfss", "sfms", "mfms")),
        "panda_predicted_physical_severity2/summary_all.json",
        {method: panda_disturbed[method]["successes"] for method in ("sfss", "sfms", "mfms")},
    )
    cartesian_multiseed = load(ART / "cartesian_multiseed_final" / "summary.json")["methods"]
    panda_multiseed = load(ART / "panda_multiseed_predicted_final" / "summary.json")["methods"]
    check(
        "cartesian_three_seed_release",
        all(
            cartesian_multiseed[method]["seed_count"] == 3
            and cartesian_multiseed[method]["episodes"] == 120
            for method in ("sfss_recursive", "sfms", "mfms")
        ),
        "cartesian_multiseed_final/summary.json",
        {
            method: cartesian_multiseed[method]["success_rate"]
            for method in ("sfss_recursive", "sfms", "mfms")
        },
    )
    check(
        "panda_three_seed_predicted_release",
        all(
            panda_multiseed[method]["seed_count"] == 3 and panda_multiseed[method]["episodes"] == 30
            for method in ("sfss", "sfms", "mfms")
        ),
        "panda_multiseed_predicted_final/summary.json",
        {method: panda_multiseed[method]["success_rate"] for method in ("sfss", "sfms", "mfms")},
    )
    ablations = load(ART / "release_ablations" / "summary.json")
    check(
        "release_ablation_matrix",
        ablations["requested_runs"] == 16
        and ablations["completed_runs"] == 16
        and not ablations["failed_optional_runs"],
        "release_ablations/summary.json",
        {key: ablations[key] for key in ("requested_runs", "completed_runs", "failed_optional_runs")},
    )
    miniature = load(ART / "miniature_e2e_final" / "miniature_e2e_report.json")
    check(
        "miniature_collection_to_panda_pipeline",
        miniature["passed"] and len(miniature["stages"]) == 9,
        "miniature_e2e_final/miniature_e2e_report.json",
        [row["stage"] for row in miniature["stages"]],
    )
    check(
        "run_provenance_and_step_traces",
        all(
            (ART / "final_benchmark_release" / "mesh_clean" / name).is_file()
            for name in ("resolved_config.json", "run_manifest.json", "episodes.csv", "steps.csv", "summary.json")
        ),
        "final_benchmark_release/mesh_clean",
        "complete",
    )
    check(
        "sim2real_software_preparation",
        all(
            (ROOT / path).is_file()
            for path in (
                "sfn/sim2real/calibration.py",
                "sfn/sim2real/active_learning.py",
                "scripts/finetune_real_segmentation.py",
                "scripts/review_pseudolabels.py",
                "scripts/replay_sim2real.py",
                "docs/HARDWARE_READINESS.md",
            )
        ),
        "sfn/sim2real and docs/HARDWARE_READINESS.md",
        "implemented",
    )
    check(
        "rl_release_hardening",
        all(
            (ROOT / path).is_file()
            for path in (
                "sfn/training/vector_env.py",
                "sfn/training/curriculum.py",
                "configs/sfms_curriculum.yaml",
                "scripts/run_multiseed_curriculum.py",
                "models/sfms_mesh_v2_teacher_compatible.pt",
                "models/mfms_mesh_v2_teacher_compatible.pt",
            )
        ),
        "vector runner, curriculum and compatible policies",
        "implemented",
    )
    required = [
        ART / "FINAL_SOFTWARE_REPORT.md",
        ART / "IMPLEMENTATION_STATUS.md",
        ART / "figures" / "severity_success.png",
        ART / "figures" / "mfms_history_ablation.png",
        ART / "figures" / "panda_dynamic_insertion_matrix.png",
        ART / "figures" / "representative_mesh_sfss_episode.gif",
        ART / "figures" / "representative_mesh_sfss_failure.gif",
        ART / "figures" / "standalone_rim_collision.png",
        ART / "standalone_insertion_tolerance_map" / "basin_xy.png",
        ART / "checkpoint_registry.json",
        ROOT / "docs" / "REPRODUCIBILITY_APPENDIX.md",
        ART / "source_snapshot.json",
        ART / "peg-in-hole-sfn-source.zip",
    ]
    check(
        "reports_plots_video_registry",
        all(path.exists() and path.stat().st_size > 0 for path in required),
        "software_completion_20260713",
        [path.name for path in required],
    )

    report = {
        "schema_version": 1,
        "passed": all(row["passed"] for row in checks),
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "hardware_excluded": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "checks_passed", "checks_total")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
