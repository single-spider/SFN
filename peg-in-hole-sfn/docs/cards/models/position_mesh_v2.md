# Model Card: position_mesh_v2

> Status: **RECORDED** | Updated: 2026-07-13

## Identity

- **Task:** `position`.
- **Checkpoint:** `models/position_mesh_v2.pt`.
- **Checkpoint SHA-256:** `80a924ed2896a546e1f77bcc209dfa95d76961e4a08b2284bb0f9d48798125a3`.
- **Size:** 5434551 bytes.
- **Checkpoint schema:** 1.
- **Model:** `PositionNet` with 450842 parameters.
- **Configuration:** `in_channels`=1, `grid_size`=21, `base`=16, `task`="position".
- **Code revision, owner, and approval:** not recorded in checkpoint metadata.

## Training provenance

- **Training dataset manifest:** `data/mesh_v2_train_randomized/manifest.json`.
- **Manifest SHA-256:** `3eb3b4b29f58c39249f4835cd7242d516ad6d8b7bb9f75597979bd594907b506`.
- **Dataset content SHA-256:** `13debc2a4c58e92abec8104baf2db919840152f33692729de23b221f87edabcd`.
- **Selected epoch and global step:** 13 and 338.
- **Training samples and selected-epoch loss:** 3252 and 2.4597324316318216.
- **Training-state scope:** optimizer and scheduler state are embedded; optimizer and scheduler names, hardware, and random seed are not recorded in the selected checkpoint metadata used by this card.

## Validation evidence

The checkpoint was selected on `mean_radial_error_mm` with mode `min`.

| Split | Backend | Samples | Metric | Result | Evidence |
|---|---|---:|---|---:|---|
| validation unseen | mesh_orthographic | 1054 | mean_radial_error_mm | 0.9414283350345746 | embedded checkpoint metadata |

Other scalar validation results: `exact_cell_accuracy`=0.25616698292220114, `within_1_cell_accuracy`=0.9098671726755219, `within_2_cell_accuracy`=0.9981024667931688, `within_5_cell_accuracy`=1.0, `mean_abs_x_mm`=0.5872865275142315, `mean_abs_y_mm`=0.5199240986717267.

## Intended use and limitations

This checkpoint supports research evaluation on the recorded mesh-orthographic input contract. The evidence does not establish performance on real cameras, real hardware, contact insertion, or shapes outside the listed manifests. Validation uncertainty intervals and independent repeat runs are not recorded. Deployment in physical robot control requires separate calibration, safety, and hardware validation.

## Change record

Recorded from the selected checkpoint, its terminal summary when present, and the linked dataset manifest on 2026-07-13. No reviewer or approval is recorded.
