# Model Card: orientation_mesh_v2_learned

> Status: **RECORDED** | Updated: 2026-07-13

## Identity

- **Task:** `orientation`.
- **Checkpoint:** `models/orientation_mesh_v2_learned.pt`.
- **Checkpoint SHA-256:** `4e140f19f11439a1ab604f3c1b3cef4d8eeea437c30e09fcb9188d0250348052`.
- **Size:** 1562081 bytes.
- **Checkpoint schema:** 1.
- **Model:** `OrientationNet` with 127503 parameters.
- **Configuration:** `in_channels`=1, `angles`=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], `base`=16, `task`="orientation".
- **Code revision, owner, and approval:** not recorded in checkpoint metadata.

## Training provenance

- **Training dataset manifest:** `data/mesh_v2_train_randomized/manifest.json`.
- **Manifest SHA-256:** `3eb3b4b29f58c39249f4835cd7242d516ad6d8b7bb9f75597979bd594907b506`.
- **Dataset content SHA-256:** `13debc2a4c58e92abec8104baf2db919840152f33692729de23b221f87edabcd`.
- **Selected epoch and global step:** 6 and 156.
- **Training samples and selected-epoch loss:** 3252 and 2.2244825546558085.
- **Training-state scope:** optimizer and scheduler state are embedded; optimizer and scheduler names, hardware, and random seed are not recorded in the selected checkpoint metadata used by this card.

## Validation evidence

The checkpoint was selected on `mean_abs_error_deg` with mode `min`.

| Split | Backend | Samples | Metric | Result | Evidence |
|---|---|---:|---|---:|---|
| validation unseen | mesh_orthographic | 1054 | mean_abs_error_deg | 4.815939278937382 | embedded checkpoint metadata |

Other scalar validation results: `exact_candidate_accuracy`=0.12144212523719165, `within_2_deg_accuracy`=0.32922201138519924, `within_4_deg_accuracy`=0.5313092979127134.

## Intended use and limitations

This checkpoint supports research evaluation on the recorded mesh-orthographic input contract. The evidence does not establish performance on real cameras, real hardware, contact insertion, or shapes outside the listed manifests. Validation uncertainty intervals and independent repeat runs are not recorded. Deployment in physical robot control requires separate calibration, safety, and hardware validation.

## Change record

Recorded from the selected checkpoint, its terminal summary when present, and the linked dataset manifest on 2026-07-13. No reviewer or approval is recorded.
