# Model Card: MFMS Mesh-v2 Teacher

> Status: **SELECTED** | Updated: 2026-07-13

- Runtime checkpoint: `models/mfms_mesh_v2_teacher_compatible.pt`.
- SHA-256: `72628918237d6dda21983fa95b76432bf8971cf60620ca199ff36e7d09c6d9ec`.
- Historical weight source: `models/mfms_mesh_v2_teacher.pt`, SHA-256 `426d53d07eec8bd6500a34f2ad1a042097e2b6b5445e00d9c6a2105809322f45`. The runtime copy has identical weights plus explicit mesh-renderer, predicted-mask and VSN hash compatibility metadata.
- Architecture: 452-dimensional input, 256-dimensional projection and recurrent hidden state, three actions, history length 4.
- Training: SFMS teacher imitation, 4,096 samples, 20 epochs, 81,920 optimizer examples.

## Fixed-candidate evaluation

The teacher-imitation candidate and history length were frozen before the final held-out evaluation; test outcomes are evidence, not a selection criterion. It achieved 40/40 clean held-out mesh insertion trials in the historical benchmark. A separate burst-occlusion ablation showed that history lengths 2, 4 and 8 performed worse than history 1; therefore this card does not claim a demonstrated temporal advantage.

The model is simulation research software and is not approved for unguarded hardware motion.
