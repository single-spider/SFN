# Model Card: position_mesh_v2_b32

> Status: **RECORDED** | Updated: 2026-07-13

## Identity

- **Task:** `position`.
- **Checkpoint:** `models/position_mesh_v2_b32.pt`.
- **Checkpoint SHA-256:** `4cfdf5c816eef3bf19e6e052d2ea2438f49c03b1077da1dc359312cc0aa07b85`.
- **Size:** 21516623 bytes.
- **Checkpoint schema:** 1.
- **Model:** `PositionNet` with 1790986 parameters.
- **Configuration:** `in_channels`=1, `grid_size`=21, `base`=32, `task`="position".
- **Code revision, owner, and approval:** not recorded in checkpoint metadata.

## Training provenance

- **Training dataset manifest:** `data/mesh_v2_train_randomized/manifest.json`.
- **Manifest SHA-256:** `3eb3b4b29f58c39249f4835cd7242d516ad6d8b7bb9f75597979bd594907b506`.
- **Dataset content SHA-256:** `13debc2a4c58e92abec8104baf2db919840152f33692729de23b221f87edabcd`.
- **Selected epoch and global step:** 28 and 728.
- **Training samples and selected-epoch loss:** 3252 and 1.2691295467890227.
- **Training-state scope:** optimizer and scheduler state are embedded; optimizer and scheduler names, hardware, and random seed are not recorded in the selected checkpoint metadata used by this card.

## Validation evidence

The checkpoint was selected on `mean_radial_error_mm` with mode `min`.

| Split | Backend | Samples | Metric | Result | Evidence |
|---|---|---:|---|---:|---|
| validation unseen | mesh_orthographic | 1054 | mean_radial_error_mm | 0.5764784334292485 | embedded checkpoint metadata |

Other scalar validation results: `exact_cell_accuracy`=0.47058823529411764, `within_1_cell_accuracy`=0.9943074003795066, `within_2_cell_accuracy`=1.0, `within_5_cell_accuracy`=1.0, `mean_abs_x_mm`=0.3415559772296015, `mean_abs_y_mm`=0.29506641366223907.

## Intended use and limitations

This checkpoint supports research evaluation on the recorded mesh-orthographic input contract. The evidence does not establish performance on real cameras, real hardware, contact insertion, or shapes outside the listed manifests. Validation uncertainty intervals and independent repeat runs are not recorded. Deployment in physical robot control requires separate calibration, safety, and hardware validation.

## Change record

Recorded from the selected checkpoint, its terminal summary when present, and the linked dataset manifest on 2026-07-13. No reviewer or approval is recorded.
