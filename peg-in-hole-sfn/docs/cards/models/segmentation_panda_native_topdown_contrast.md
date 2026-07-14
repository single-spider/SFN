# Model Card: Panda Native Top-Down Contrast Segmentation

> Status: **SELECTED FOR SIMULATION** | Updated: 2026-07-13

## Identity and training

- Checkpoint: `models/segmentation_panda_native_topdown_contrast.pt`.
- SHA-256: `81d746742f3e30eb4386237944c242a90f9c66444d9d8fa47c304a1aa3c254d6`.
- Architecture: three-class segmentation network, base width 16.
- Training data: 1,716 Panda-native high-contrast frames.
- Selection data: 542 validation-unseen frames.
- Selected epoch/global step: 12 / 5,148.
- Selection metric: validation mean IoU.

## Evidence

| Split | Samples | Mean IoU | Background IoU | Peg IoU | Seam IoU |
|---|---:|---:|---:|---:|---:|
| validation unseen | 542 | 0.9980 | 0.9999 | 0.9954 | 0.9987 |
| test unseen | 542 | 0.9989 | 0.9999 | 0.9980 | 0.9989 |

The predicted-mask mesh-template pose evaluation on the test split gives 0.2609 mm mean XY error and 1.6212° mean yaw error.

## Intended use and limitations

The model is selected only for the disclosed high-contrast PyBullet camera domain. It is not approved for real-camera or safety-critical control. Hardware use requires real-image adaptation, calibration, uncertainty checks and guarded motion.

