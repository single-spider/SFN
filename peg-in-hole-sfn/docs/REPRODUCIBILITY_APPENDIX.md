# Reproducibility Appendix

## Scope

This appendix covers the supported simulation release. It does not cover physical Panda communication, measured hand–eye calibration, manufactured-part metrology or real insertion trials.

## Environment

- Supported Python: 3.11–3.12.
- Project installation: editable installation from `pyproject.toml` with the `dev` extra.
- CPU release gate: `scripts/smoke_software.py --full`.
- CUDA parity gate: `scripts/gpu_smoke.py`.
- CI definition: `.github/workflows/cpu-ci.yml`.

The release evidence stores Python, platform, package versions, source revision and dirty-state information in each `run_manifest.json`. The final source inventory is stored in `artifacts/software_completion_20260713/source_snapshot.json`.

## Immutable inputs

Selected datasets and checkpoints are indexed by SHA-256 in:

- `artifacts/software_completion_20260713/checkpoint_registry.json`;
- `docs/cards/datasets`;
- `docs/cards/models`.

Runtime-compatible SFMS and MFMS policy copies contain the expected renderer, mask source and VSN checkpoint hashes. The evaluator rejects missing or mismatched compatibility metadata unless the diagnostic override is explicitly requested.

## Small reconstruction

The complete lightweight path is:

```powershell
python scripts/run_miniature_e2e.py --out artifacts/miniature_e2e
```

It performs dataset collection, one-epoch segmentation/position/orientation training, SFMS and MFMS teacher warm starts, Cartesian evaluation, Panda evaluation and report generation. Completion is recorded in `miniature_e2e_report.json`.

## Final benchmark reconstruction

The short validation form is:

```powershell
python scripts/run_final_benchmark.py --quick --out artifacts/final_benchmark_quick
```

The evidence-sized form removes `--quick`. The benchmark manifest records the full command graph and input hashes. Each controller evaluation directory contains:

- `resolved_config.json`;
- `run_manifest.json`;
- `episodes.csv`;
- `steps.csv`;
- `summary.json`;
- per-shape summaries where applicable.

The summary files report trial counts, Wilson success intervals and distributions for steps, position error, yaw error and latency. Paired method comparisons use episode identity keys, paired bootstrap intervals and exact McNemar tests.

## Long multi-seed training

The final curriculum implementation supports independent seeds, synchronous environments, fixed validation seeds, periodic checkpoints and JSONL metrics. A representative three-seed launch is:

```powershell
python scripts/run_multiseed_curriculum.py --seeds 101,102,103 --segmentation models/segmentation_mesh_v2_ce.pt --position models/position_mesh_v2_geometric.pt --orientation models/orientation_mesh_v2_symmetry_hybrid.pt --initial-policy models/sfms_mesh_v2_teacher_compatible.pt --out artifacts/sfms_multiseed
```

This is a long experiment and is not required to reproduce the already selected teacher-policy results. Any new multi-seed result must be treated as a new model-selection experiment and documented in a separate model card.

## Evidence interpretation

Results must retain their backend labels:

- `toy_direct`: simplified legacy Cartesian reference;
- `mesh_orthographic`: asset-faithful Cartesian visual pipeline;
- `standalone_pybullet_raster_compound`: standalone physical insertion;
- `panda_kinematic`: idealized coordinate and IK validation;
- `panda_dynamic` / `panda_native_camera`: simulated motor, camera and insertion execution;
- `real_hardware`: reserved for measurements that do not yet exist.

An artifact from one backend must not be used as evidence for another.
