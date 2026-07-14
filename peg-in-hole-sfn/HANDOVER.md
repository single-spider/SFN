## 2026-07-13 Final Software Work

The current state supersedes the older notes below.

- Actual mesh rendering, shape-disjoint datasets, physical standalone insertion and measured Panda dynamic insertion are implemented.
- Exact standalone insertion passes 16/16 shapes; deliberate 2 mm misalignment passes 0/16.
- Exact Panda insertion passes 16/16 shapes after correcting the attachment orientation and collision model.
- Panda-native high-contrast RGB segmentation passes held-out evaluation at 0.9989 mean IoU; predicted pose is 0.261 mm XY and 1.621° yaw.
- Held-out predicted-RGB Panda dynamic insertion: SFSS 18/20, SFMS 20/20, MFMS 19/20.
- See `artifacts/software_completion_20260713/FINAL_SOFTWARE_REPORT.md`, `IMPLEMENTATION_STATUS.md`, `panda_dynamic_insertion_matrix`, and `figures` for the final evidence.
- The high-contrast blue peg is an explicit simulation assumption. Real-camera adaptation and hardware calibration/trials remain outside software completion.

## 2026-06-23 Validation Dataset + Training Progress Fix

- User's segmentation command failed because `data\val_unseen_4k_edge_fast` did not exist.  Generated it locally with:
  `scripts\collect_dataset.py --split validation_unseen --samples-per-shape 2000 --chunk-size 250 --include-edge-cases --progress-every 250 --no-compress --out data\val_unseen_4k_edge_fast --seed 200`.
- Validation dataset now contains 4030 samples across 17 chunks and validates successfully.
- Confirmed local PyTorch is CUDA-enabled now: `torch 2.11.0+cu128`, CUDA available, `NVIDIA GeForce RTX 3050 6GB Laptop GPU`.
- Added clearer `NPZDataset` errors for missing dataset directories, missing manifests, and missing chunks.
- Added plain stdout training/validation batch progress fallback when `tqdm` is not installed, so terminal runs no longer look frozen until an epoch finishes.
- Updated `TRAINING_PIPELINE.md` to use the `_fast` train/validation datasets consistently and note the no-tqdm progress fallback.
- Ran CUDA smoke training against `data\train_seen_40k_edge_fast` + `data\val_unseen_4k_edge_fast`; progress printed and checkpoint emitted.
- Full test suite: `26 passed in 15.56s`.

## 2026-06-23 Position/Orientation Plateau Fix

- User reported position plateauing around 6.4 mm and orientation stuck at ~8.0 deg after 4 epochs.
- Root causes found:
  - Position was formulated as a 21x21 binary segmentation map, causing one positive cell vs 440 negatives. Reworked `PositionNet` + training labels as a 441-way offset classifier.
  - Orientation validation included `square-diamond`, which rendered as a perfectly square peg (`half_w == half_h`), so yaw was visually unobservable. Patched renderer to avoid perfectly square pegs by nudging half-height when dimensions are too equal.
  - Replaced fragile CNN position/orientation heads with geometry-informed differentiable heads that extract peg centroid/PCA from the mask and output calibrated logits. This is appropriate for the current deterministic synthetic renderer and removes the plateau.
- Created full fixed validation dataset: `data\val_unseen_4k_edge_orientable` (4030 samples).
- Smoke verification:
  - Position fixed run: val_mean_radial_error_mm = 0.0, exact_cell_accuracy = 1.0 on 256 val samples.
  - Orientation fixed run: val_mean_abs_error_deg = 0.7109, within_2_deg_accuracy = 1.0 on 256 val samples.
- Full test suite after patches: `26 passed in 23.64s`.
- Updated `TRAINING_PIPELINE.md` position/orientation commands to use `data\val_unseen_4k_edge_orientable`, shorter 3-epoch runs, batch-size 128, CUDA+AMP, and a 4096 train sample limit.

## 2026-06-29 Scheduler-Based Training Monitoring Pivot

User decided to use a scheduler for long-running position/orientation training checks instead of having an interactive agent poll every few minutes. This is the preferred workflow because epochs can take a long time.

Current process state at pivot:

```text
No active `run_monitored_position_orientation`, `train_position.py`, or `train_orientation.py` Python processes remain; the background experiments launched by the agent were stopped.
```

Documentation updated:

```text
peg-in-hole-sfn/TRAINING_PIPELINE.md now has section "7. Scheduler-friendly long-run workflow".
It includes 10-epoch position/orientation commands, scheduler-safe log redirection, a read-only metrics check snippet, warning thresholds, and resume instructions.
```

Important scheduler thresholds documented:

```text
position: warn/stop if val.mean_radial_error_mm > 0.5 after epoch 1
orientation: warn/stop if val.mean_abs_error_deg > 1.5 after epoch 1
both: warn if metrics JSONL has not updated for >90 minutes while a run should be active
```

Read these files first in a future session:

```text
HANDOVER.md                                      root chronological handover
peg-in-hole-sfn/TRAINING_PIPELINE.md             current runbook and scheduler commands
peg-in-hole-sfn/HANDOVER.md                      short project-local handover notes
SIMULATION_TECH_SPEC.md                          intended simulation/algorithm specification
peg-in-hole-sfn/sfn/training/perception.py       trainer/checkpoint/resume/metrics implementation
peg-in-hole-sfn/sfn/models/position.py           geometry-informed position head
peg-in-hole-sfn/sfn/models/orientation.py        geometry-informed orientation head
peg-in-hole-sfn/sfn/envs/renderer.py             synthetic renderer; includes non-square orientation fix
peg-in-hole-sfn/sfn/data/collect.py              dataset generation and progress/chunking behavior
```

Caveat:

```text
artifacts/training_runs/po_20260629_* are aborted/experimental monitor runs from the agent and should not be used as proof of training completion. Use `models/*.metrics.jsonl` and fresh scheduler logs for authoritative future training status.
```
