# Model Card: SFMS Mesh-v2 Teacher

> Status: **REFERENCE BASELINE** | Updated: 2026-07-13

- Runtime checkpoint: `models/sfms_mesh_v2_teacher_compatible.pt`.
- SHA-256: `7120f480c00eb2d95b986d9119f88001daf93f10b2794dbe98384cab66dd1b2a`.
- Historical weight source: `models/sfms_mesh_v2_teacher.pt`, SHA-256 `c0d9cc7e34bc3ddc520840882a53aeef5a5c27dd71d3aa3b8a048f7adf84d7b0`. The runtime copy has identical weights plus explicit mesh-renderer, predicted-mask and VSN hash compatibility metadata.
- Architecture: 452-dimensional VSN state, hidden layers 256/128, three actions.
- Training: teacher imitation, 4,096 samples, 20 epochs, 81,920 optimizer examples.
- Intended input: the position/orientation probability contract, not ground-truth pose.

## Evaluation evidence

The checkpoint achieved 40/40 clean held-out mesh insertion trials in the historical comparison. Those test results are retained as baseline evidence and are not used for current model selection. A locked 200-episode `validation_unseen` comparison selected the RL candidate instead (193/200 versus 190/200); see `artifacts/software_completion_20260713/policy_selection_validation_200/`.

The model is simulation research software. Hardware execution requires independent camera, calibration, dynamics and safety validation.
