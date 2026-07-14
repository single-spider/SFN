# Model Card: segmentation_mesh_v2_ce

> Status: **RECORDED** | Updated: 2026-07-13

## Identity

- **Task:** `segmentation`.
- **Checkpoint:** `models/segmentation_mesh_v2_ce.pt`.
- **Checkpoint SHA-256:** `ba8edd4ee5bf8e7dc9d2fac55e44f02545ada33ab68b3742f8092033ed49fda9`.
- **Size:** 46417 bytes.
- **Checkpoint schema:** 1.
- **Model:** `SegmentationModel` with 2819 parameters.
- **Configuration:** `in_channels`=3, `classes`=3, `base`=16, `task`="segmentation".
- **Code revision, owner, and approval:** not recorded in checkpoint metadata.

## Training provenance

- **Training dataset manifest:** `data/mesh_v2_train_randomized/manifest.json`.
- **Manifest SHA-256:** `3eb3b4b29f58c39249f4835cd7242d516ad6d8b7bb9f75597979bd594907b506`.
- **Dataset content SHA-256:** `13debc2a4c58e92abec8104baf2db919840152f33692729de23b221f87edabcd`.
- **Selected epoch and global step:** 11 and 561.
- **Training samples and selected-epoch loss:** 3252 and 0.04815308486714082.
- **Training-state scope:** optimizer and scheduler state are embedded; optimizer and scheduler names, hardware, and random seed are not recorded in the selected checkpoint metadata used by this card.

## Validation evidence

The checkpoint was selected on `mean_iou` with mode `max`.

| Split | Backend | Samples | Metric | Result | Evidence |
|---|---|---:|---|---:|---|
| validation unseen | mesh_orthographic | 1054 | mean_iou | 0.9140346088547346 | embedded checkpoint metadata |

Other scalar validation results: `pixel_accuracy`=0.989516679316888, `class_iou`={'0': 0.9889121167749658, '1': 0.9680172168286997, '2': 0.7851744929605387}.

## Intended use and limitations

This checkpoint supports research evaluation on the recorded mesh-orthographic input contract. The evidence does not establish performance on real cameras, real hardware, contact insertion, or shapes outside the listed manifests. Validation uncertainty intervals and independent repeat runs are not recorded. Deployment in physical robot control requires separate calibration, safety, and hardware validation.

## Change record

Recorded from the selected checkpoint, its terminal summary when present, and the linked dataset manifest on 2026-07-13. No reviewer or approval is recorded.
