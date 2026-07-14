# Dataset Card: Panda Native Top-Down Contrast

> Status: **RECORDED (LEGACY CONTRACT)** | Collected: 2026-07-13 | Schema: 1

## Identity

| Split | Samples | Shapes | Seed | Manifest SHA-256 |
|---|---:|---:|---:|---|
| train seen | 1,716 | 12 | 3201 | `5f0cf36a4a30b6086c5b6bb635ac274891b165f890c0a43d48edcc48dd4adb5d` |
| validation unseen | 542 | 2 | 3202 | `2d726c0f2d866c0ad94e283710584948f65f0e13cf5809d544a8150a2359aa90` |
| test unseen | 542 | 2 | 3203 | `5028298e49468405527d5aa307908b4c2785275462a66cd71d76f2757678451c` |

The manifests are stored under `data/panda_native_topdown_contrast_{train,validation,test}`. The split shape lists are disjoint. These recorded artifacts predate the dataset-v2 collector contract: their manifests truthfully declare schema 1 and their chunks do not contain the v2 camera-variant, augmentation, family/symmetry, or episode/frame arrays. Regenerate them with the current collector before treating them as schema-v2 data.

## Observation contract

- Backend: PyBullet Panda native RGB and simulator semantic body IDs.
- Camera: fixed top-down, 500 × 400 pixels, 35° vertical field of view.
- Workpiece: high-contrast blue simulated peg; neutral fixture and Panda hand.
- Labels: background 0, peg 1, visible hole seam 2.
- Pose labels: measured Panda task-frame X/Y/yaw error.
- Edge cases: workspace boundary and strict-tolerance examples are included.

## Intended use and limitations

This dataset tests the end-to-end software pipeline when the workpiece is visually distinguishable from the robot. It is not a real-camera dataset and does not establish sim-to-real segmentation. Materials, reflections, calibration error, motion blur and real lighting must be measured or represented before hardware deployment.


