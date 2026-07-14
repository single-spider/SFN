# Model Card: SFMS Mesh-v2 Validation-Selected RL

> Status: **SELECTED** | Updated: 2026-07-13

- Runtime checkpoint: `models/sfms_mesh_v2_rl_best_compatible.pt`.
- SHA-256: `04ac68aa9dcc5bb782e810d298f8100dcaae73a71066bff5057ab59eac3c1d5b`.
- Architecture: 452-dimensional VSN probability state, hidden layers 256/128, three bounded actions.
- Training: teacher-imitation warm start followed by stabilized A2C fine-tuning.
- Intended input: predicted-mask VSN probabilities from the mesh-orthographic camera contract; no ground-truth pose or robot state.

## Validation-only selection

The candidate was selected before final retesting using 200 paired `validation_unseen` insertion episodes at seed 9410. It achieved 193/200 successes, 0.152 mm mean final XY error and 0.517 degree mean final yaw error. The teacher reference achieved 190/200, 0.179 mm and 0.497 degree on the same schedule. Raw records, resolved configurations and manifests are in `artifacts/software_completion_20260713/policy_selection_validation_200/`.

## Final evaluation rule

Held-out test results are reported once after this selection and are not used to switch back to another checkpoint. This prevents test-shape model selection. The checkpoint remains simulation research software; real execution requires camera, calibration, dynamics and safety validation.
