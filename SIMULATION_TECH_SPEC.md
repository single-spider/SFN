# SFN Complete Simulation Technical Specification

**Status:** Implementation specification only  
**Source context:** `HANDOVER.md`, current repository code, and `2204.07776v2.pdf`  
**Primary objective:** Complete the SFN simulation, training, evaluation, and reproducibility pipeline without implementing any real robot, camera, force/torque sensor, or robot middleware integration.

---

## 1. Scope

The finished project shall support this complete simulated pipeline:

```text
simulated RGB image
-> simulated or predicted three-class segmentation mask
-> position and orientation heatmaps
-> SFSS, SFMS, or MFMS controller
-> simulated dx/dy/yaw motion
-> optional simulated z insertion
-> success/failure, metrics, artifacts, and reproducible reports
```

### 1.1 Required deliverables

1. A deterministic, testable simulation environment.
2. A contact-free alignment task and a standalone-object insertion task.
3. Dataset generation and shape-disjoint dataset splits.
4. Trainable segmentation, position, and orientation models.
5. A unified virtual sensor network (VSN) inference API.
6. SFSS closed-loop evaluation.
7. SFMS reinforcement-learning training and evaluation.
8. MFMS recurrent reinforcement-learning training and evaluation.
9. Seen/unseen shape evaluation, ablations, robustness tests, and reports.
10. Automated smoke, unit, integration, and regression tests.
11. Configuration files, checkpoints, logs, and reproducible command-line entry points.

### 1.2 Explicitly out of scope

Do **not** implement:

- UR5, Franka, or other physical robot communication.
- ROS/ROS2 nodes, MoveIt, RTDE, vendor SDKs, or PLC interfaces.
- Real camera drivers or hardware calibration capture.
- Real force/torque sensor drivers.
- Real-world data collection or automatic real-world mask annotation.
- Safety PLC logic, emergency stops, physical collision limits, or operator UI.
- Deployment of commands to physical hardware.

Interfaces may be defined for later hardware integration, but they must remain unimplemented abstractions.

### 1.3 Definition of “complete simulation”

Completion does not require a dynamically controlled robot arm. The simulation shall move a standalone peg in Cartesian coordinates. This avoids unstable inverse kinematics while preserving the paper’s visual alignment problem.

Two task modes are required:

1. **Alignment mode:** success is determined by residual XY and yaw error.
2. **Insertion mode:** after alignment, the standalone peg descends in Z and succeeds only if it reaches the configured insertion depth without invalid collision.

The Panda-based `peg_in_hole_v11.py` remains a legacy reference and visual aid. It shall not be the foundation of the completed pipeline.

---

## 2. Current-State Findings

The implementation must account for these defects in the current repository:

1. `peg_in_hole_v11.py` declares an observation space that does not match its dictionary observation.
2. Its action space is `Discrete(4)` although `step()` consumes `[dx, dy, dyaw]`.
3. Training and evaluation behavior are coupled to `test_mode`.
4. In training mode, the supplied action is ignored and a random pose is sampled.
5. `step()` always returns reward `0`, `done=False`, and empty info.
6. It uses the old four-value Gym step API.
7. Its reset observation reports zero pose error even though later error is measured from PyBullet link state.
8. Rendering and mask generation rely on current working-directory-relative paths.
9. Pyrender renderers are created without guaranteed cleanup.
10. Global NumPy and Python RNG state is mutated instead of using per-environment generators.
11. Position and orientation optimizer loops are commented out.
12. The existing training split assumes the final environment is a test environment and fails for one environment.
13. Model checkpoints serialize complete Python objects rather than stable state dictionaries.
14. Inference scripts use ground-truth masks and duplicate decoding logic.
15. The real segmentation dataloader is missing.
16. Existing A2C variants are generic or experimental and are not cleanly wired to `[position heatmap, orientation heatmap]`.
17. No automated test suite verifies assets, coordinate signs, reward, termination, or model interfaces.

New code must be added in a clean package instead of progressively adding more behavior to legacy experimental scripts.

---

## 3. Target Repository Layout

Create the following structure under `peg-in-hole-sfn/`:

```text
peg-in-hole-sfn/
  sfn/
    __init__.py
    config.py
    constants.py
    geometry.py
    seeding.py

    envs/
      __init__.py
      asset_registry.py
      scene.py
      renderer.py
      alignment_env.py
      insertion_env.py
      wrappers.py

    data/
      __init__.py
      schema.py
      collect.py
      dataset.py
      splits.py
      augment.py
      validate.py

    models/
      __init__.py
      unet.py
      segmentation.py
      position.py
      orientation.py
      vsn.py
      controllers.py

    training/
      __init__.py
      common.py
      train_segmentation.py
      train_position.py
      train_orientation.py
      train_sfms.py
      train_mfms.py

    evaluation/
      __init__.py
      metrics.py
      evaluator.py
      evaluate_perception.py
      evaluate_sfss.py
      evaluate_sfms.py
      evaluate_mfms.py
      ablations.py
      reporting.py

  configs/
    base.yaml
    data.yaml
    segmentation.yaml
    position.yaml
    orientation.yaml
    sfss.yaml
    sfms.yaml
    mfms.yaml
    evaluation.yaml

  scripts/
    validate_assets.py
    collect_dataset.py
    train_segmentation.py
    train_position.py
    train_orientation.py
    train_sfms.py
    train_mfms.py
    evaluate.py
    run_demo.py

  tests/
    test_assets.py
    test_geometry.py
    test_env_contract.py
    test_rendering.py
    test_dataset.py
    test_position_codec.py
    test_orientation_codec.py
    test_vsn.py
    test_reward.py
    test_sfss_integration.py
    test_rl_smoke.py

  data/                 # ignored generated data
  artifacts/            # ignored run outputs
  models/               # ignored exported checkpoints
```

Legacy files may remain, but new scripts must import only from `sfn.*`.

---

## 4. Global Conventions

### 4.1 Coordinate system

Use one documented coordinate convention everywhere:

- World `+X`: camera-image left after applying the existing transform.
- World `+Y`: camera-image down after applying the existing transform.
- World `+Z`: upward.
- Positive yaw: counter-clockwise about `+Z`.
- Pose error is always:

```python
error = peg_pose - hole_pose
```

- A corrective command should approximately be the negative error.

The existing heatmap mapping must be preserved:

```python
col = round(-dx_m * 1000) + 10
row = round( dy_m * 1000) + 10
```

The inverse mapping is:

```python
dx_m = -(col - 10) / 1000
dy_m =  (row - 10) / 1000
```

Unit tests must verify all four cardinal directions.

### 4.2 Units

- Internal translation: meters.
- User-facing translation and reports: millimeters.
- Internal and user-facing yaw: degrees, except when calling PyBullet quaternion functions.
- Images: `uint8`, RGB, channel-first only at model boundaries.
- Masks: `uint8`, values `{0, 1, 2}`.
- Neural-network inputs: `float32`.

### 4.3 Segmentation classes

```text
0 = background/base surface
1 = peg
2 = visible seam or uncovered hole region
```

The seam class is the visible part of the hole not occluded by the peg.

### 4.4 Default image and heatmap sizes

```text
full render: 1280 x 720
model crop: 250 x 200
RGB tensor: [3, 200, 250]
mask tensor: [200, 250]
position heatmap: [21, 21]
orientation candidates: 11
orientation angles: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10] degrees
```

### 4.5 Shape splits

Use a shape-disjoint default split.  For the current 16 synthetic shape assets,
prefer a 75/12.5/12.5 split (12 train, 2 validation, 2 final test).  This gives
the learned perception models enough shape diversity while preserving unseen
shape validation and final holdout reporting.

```yaml
train_seen:
  - square-triangle
  - square-square
  - square-pentagon
  - square-hexagon
  - square-concave1
  - square-convex1
  - square-convex2
  - square-convex3
  - square-convex4
  - square-fillet1
  - square-fillet2
  - square-fillet3

validation_unseen:
  - square-diamond
  - square-trapezoid

test_unseen:
  - square-concave2
  - square-fillet4
```

Never randomly mix samples from one shape across a “seen versus unseen” comparison.

---

## 5. Segment 0 — Project Foundation

### Goal

Create stable configuration, seeding, logging, checkpoint, and package conventions.

### Required implementation

1. Use `pathlib.Path` and resolve all asset paths relative to the module file.
2. Use YAML configuration loaded into validated dataclasses.
3. Every run must save:
   - resolved configuration,
   - command line,
   - seed,
   - Python and package versions,
   - Git commit and dirty status when available,
   - timestamps,
   - metrics JSONL/CSV,
   - checkpoints,
   - final summary JSON.
4. Provide one seeding function covering:
   - Python `random`,
   - NumPy,
   - PyTorch CPU/CUDA,
   - environment-local RNG.
5. Checkpoints must contain:

```python
{
    "schema_version": 1,
    "model_name": str,
    "model_config": dict,
    "model_state_dict": dict,
    "optimizer_state_dict": dict | None,
    "scheduler_state_dict": dict | None,
    "epoch": int,
    "global_step": int,
    "metrics": dict,
    "data_split": dict,
}
```

Do not save complete model objects with `torch.save(model, ...)`.

### Acceptance criteria

- The same seed produces identical initial poses and masks in direct-render mode.
- A checkpoint loads on CPU even if trained on CUDA.
- No module depends on the process current working directory.

---

## 6. Segment 1 — Asset Registry and Validation

### Goal

Make all shape assets discoverable and fail early when malformed.

### `AssetRegistry`

Required public interface:

```python
class AssetRegistry:
    def list_shapes(self) -> list[str]: ...
    def get(self, shape: str) -> ShapeAssets: ...
    def validate(self, shape: str) -> AssetValidationResult: ...
    def validate_all(self) -> dict[str, AssetValidationResult]: ...
```

`ShapeAssets` must contain absolute paths for:

```text
base/base.urdf
base/base.obj
peg/peg.obj
peg/peg_test.urdf
mask.obj
```

`peg/peg.urdf` is optional for the new standalone-object environment.

Validation must check:

1. Required files exist.
2. URDFs load in a temporary PyBullet DIRECT connection.
3. Mesh bounds are finite and non-empty.
4. Base and peg scale are compatible.
5. `mask.obj` can be loaded by Trimesh.
6. Standalone peg has visual and collision geometry.
7. No asset name silently falls back to another shape.

### Acceptance criteria

- All 16 current shape directories pass validation or are explicitly listed as invalid with actionable errors.
- `scripts/validate_assets.py` exits nonzero if any requested asset is invalid.

---

## 7. Segment 2 — Deterministic Alignment Environment

### Goal

Replace the overloaded legacy environment with a correct Cartesian peg/hole environment.

### 7.1 Environment class

Create:

```python
class PegInHoleAlignmentEnv(gym.Env):
    ...
```

Use Gym 0.26 semantics:

```python
obs, info = env.reset(seed=seed, options=options)
obs, reward, terminated, truncated, info = env.step(action)
```

### 7.2 Scene model

Load:

- fixed standalone base,
- fixed or kinematic standalone peg,
- no articulated robot.

The peg is moved with `resetBasePositionAndOrientation()` or a deterministic kinematic equivalent. Physics stepping may be used for collision queries, but must not introduce uncontrolled arm dynamics.

### 7.3 Action space

Use normalized continuous actions:

```python
spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

Map to per-step deltas:

```text
dx:   ±2.0 mm default
dy:   ±2.0 mm default
dyaw: ±2.0 deg default
```

All limits must be configurable.

Actions are **incremental corrections**, not absolute poses.

Clip accumulated pose to:

```text
X/Y workspace: ±15 mm default
yaw workspace: ±15 deg default
```

### 7.4 Observation space

Use `spaces.Dict`:

```python
{
    "rgb":         Box(0, 255, shape=(3, 200, 250), dtype=uint8),
    "mask":        Box(0, 2, shape=(200, 250), dtype=uint8),
    "pose_error":  Box(..., shape=(3,), dtype=float32),
}
```

`pose_error` is `[dx_m, dy_m, dyaw_deg]`.

Ground-truth state is included for training and metrics. Wrappers must be able to hide it from policies.

### 7.5 Reset contract

`reset(options=...)` supports:

```python
{
    "shape": str | None,
    "pose_error": [dx_m, dy_m, dyaw_deg] | None,
    "error_range": {
        "xy_m": [min, max],
        "yaw_deg": [min, max],
    } | None,
}
```

Default random initialization:

```text
dx, dy uniformly in [-10, 10] mm
dyaw uniformly in [-10, 10] deg
```

Reject samples inside the success region when the caller requests a nontrivial episode.

### 7.6 Success and termination

Default alignment success:

```text
abs(dx) <= 1.0 mm
abs(dy) <= 1.0 mm
abs(dyaw) <= 2.0 deg
```

Also report radial XY error.

```text
terminated = success or unrecoverable/out-of-bounds failure
truncated = step_count >= max_steps
```

Default `max_steps=20`.

### 7.7 Reward

Provide two selectable reward modes.

#### Practical dense reward — default

Normalize error:

```python
e_xy = hypot(dx_mm, dy_mm) / xy_range_mm
e_yaw = abs(dyaw_deg) / yaw_range_deg
E = w_xy * e_xy + w_yaw * e_yaw
```

Default:

```text
w_xy = 0.7
w_yaw = 0.3
step_penalty = 0.01
success_bonus = 1.0
failure_penalty = 1.0
```

Reward:

```python
reward = previous_E - current_E - step_penalty
if success:
    reward += success_bonus
if out_of_bounds:
    reward -= failure_penalty
```

#### Paper-compatible reward

Support a configurable form based on paper equation 10:

```text
success: 1 - alpha * position_loss - beta * orientation_loss
otherwise: -1/k_max - alpha * position_loss - beta * orientation_loss
```

This mode is for reproduction experiments. The practical dense reward is the required default because it is well-defined before learned losses are available.

### 7.8 Info dictionary

Every step and reset must report:

```python
{
    "shape": str,
    "seed": int,
    "step": int,
    "pose_error": np.ndarray[3],
    "xy_error_mm": float,
    "yaw_error_deg": float,
    "success": bool,
    "out_of_bounds": bool,
    "action_normalized": np.ndarray[3] | None,
    "action_physical": np.ndarray[3] | None,
}
```

### Acceptance criteria

- The declared spaces contain every emitted observation/action.
- Applying the exact negative pose error, within action limits over multiple steps, converges.
- The oracle controller succeeds in at least 99/100 deterministic episodes for every valid shape.
- No episode exceeds `max_steps`.
- Two environments with different seeds do not share RNG state.

---

## 8. Segment 3 — Rendering and Ground-Truth Masks

### Goal

Produce synchronized RGB and masks without color-threshold hacks.

### 8.1 Preferred rendering path

Use PyBullet object/segmentation IDs from `getCameraImage()`:

1. Render RGB.
2. Render or retrieve object IDs.
3. Identify peg pixels from peg body ID.
4. Obtain the full hole-opening mask from `mask.obj` using Pyrender or a dedicated hidden render object.
5. Compute:

```python
seam_mask = hole_opening_mask AND NOT peg_mask
```

6. Compose class mask.

Do not infer peg pixels from “green” RGB thresholds.

### 8.2 Camera

Represent camera parameters with a dataclass:

```python
CameraConfig(
    width=1280,
    height=720,
    crop_width=250,
    crop_height=200,
    fov_y_deg=45.0,
    near=0.001,
    far=10.0,
    eye_offset=(0.0, 0.1, 0.1),
    up=(0.0, -1.0, 0.0),
)
```

Store view/projection matrices in dataset metadata.

### 8.3 Domain randomization

Configurable training-only randomization:

- light direction/intensity,
- peg/base colors,
- background color or texture,
- RGB brightness/contrast/gamma,
- Gaussian sensor noise,
- blur,
- crop translation up to a configured pixel range,
- camera eye/target perturbation,
- small focal/FOV perturbation,
- synthetic occluders.

Randomization must never alter the ground-truth pose label.

Provide levels:

```text
none
light
medium
heavy
```

Validation and test default to `none`, unless running a robustness suite.

### 8.4 Resource lifecycle

- Reuse renderer objects across frames.
- Explicitly delete Pyrender offscreen renderers on close.
- Disconnect only the environment’s own PyBullet client ID.
- Pass `physicsClientId` to PyBullet calls where supported.

### Acceptance criteria

- Mask values are only `0`, `1`, and `2`.
- RGB and mask crops are always aligned.
- Repeated rendering does not increase renderer or PyBullet connection counts.
- A 1,000-frame render smoke test completes without resource exhaustion.

---

## 9. Segment 4 — Standalone Insertion Environment

### Goal

Provide simulated insertion success without an articulated robot.

Create:

```python
class PegInHoleInsertionEnv(PegInHoleAlignmentEnv):
    ...
```

### 9.1 Episode phases

```text
ALIGN -> DESCEND -> SUCCESS or FAILURE
```

Default agent controls only `[dx, dy, dyaw]`.

When alignment success is reached:

1. Enter `DESCEND`.
2. Move the peg down in deterministic increments.
3. Query contacts/collision penetration.
4. Stop on target insertion depth or invalid collision.

Optional advanced configuration may expose Z as a fourth action, but it is not required for SFN reproduction.

### 9.2 Insertion success

Success requires all:

1. Peg reaches target insertion depth.
2. Residual XY/yaw remains within configured insertion tolerance.
3. No collision impulse or penetration exceeds configured failure thresholds.
4. Peg does not leave the workspace.

Default descent:

```text
descent increment: 0.25 mm
target depth: asset-relative, default 8 mm
maximum descent attempts: 64
```

Because asset mesh geometry may differ, compute and validate an asset-specific safe depth during asset validation. Store this in generated metadata or a checked-in asset manifest.

### 9.3 Collision behavior

Provide two modes:

1. `geometric`: use collision/contact queries while kinematically moving the peg.
2. `proxy`: insertion succeeds from pose tolerances only.

`geometric` is required for final completion; `proxy` is allowed for early segments and debugging.

### Acceptance criteria

- Exact alignment inserts successfully for every valid asset.
- A configured misalignment beyond tolerance fails insertion.
- Success/failure is deterministic with fixed seed and pose.
- No Panda IK is used.

---

## 10. Segment 5 — Dataset Pipeline

### Goal

Generate reusable datasets instead of rendering inside every supervised training step.

### 10.1 Dataset record

Each sample must provide:

```python
{
    "rgb": np.uint8[3, 200, 250],
    "mask": np.uint8[200, 250],
    "pose_error": np.float32[3],
    "position_target": np.uint8[21, 21],
    "orientation_index": np.int64,
    "orientation_angle_deg": np.float32,
    "shape_id": str,
    "sample_id": int,
    "seed": int,
    "camera_variant": int,
    "domain_randomization": dict,
}
```

### 10.2 Label codecs

Create one canonical implementation:

```python
encode_position(dx_m, dy_m) -> row, col
decode_position(row, col) -> dx_m, dy_m
encode_orientation(dyaw_deg) -> candidate_index
decode_orientation(index) -> angle_deg
```

All collectors, trainers, evaluators, and controllers must use these functions.

### 10.3 Storage

Support:

- chunked compressed NPZ for baseline compatibility,
- a manifest JSON,
- optional memory-mapped or HDF5 backend only if needed.

Required manifest fields:

```text
schema version
creation command
date
seed
Git commit
environment config
camera config
shape split
sample counts
chunk checksums
class pixel counts
pose histograms
```

### 10.4 Sampling

Production default:

```text
10,000 samples per training shape
2,000 samples per validation shape
2,000 samples per test shape
```

Allow smaller smoke configurations.

Sample XY and yaw with stratification so every position cell and orientation candidate receives coverage. Do not rely solely on unrestricted uniform random sampling. Training subsetting must also be stratified by shape and target bin; prefix-based dataset limits are allowed only for debugging because chunked datasets are written shape-by-shape.

### 10.5 Data validation

The validator must check:

- shapes/dtypes,
- legal class values,
- no NaNs/infinities,
- target consistency with pose,
- all chunks referenced by manifest exist,
- no duplicate `(shape_id, sample_id)`,
- no train/validation/test shape leakage,
- minimum class coverage,
- minimum pose-bin coverage.

### Acceptance criteria

- A smoke dataset can be collected and loaded on CPU.
- The same seed and configuration produce identical labels.
- Dataset validation fails after intentional target corruption.

---

## 11. Segment 6 — Segmentation Network

### Goal

Train `RGB -> three-class mask` entirely in simulation.

### 11.1 Model

Use a configurable U-Net:

```text
input:  [B, 3, 200, 250]
output: [B, 3, 200, 250] logits
```

The existing U-Net may be refactored, but the new module must not import missing `dataloader.py`.

### 11.2 Preprocessing

- Convert `uint8` RGB to `float32`.
- Divide by 255.
- Optionally normalize with dataset mean/std stored in the checkpoint.
- Never convert RGB to BGR implicitly.

### 11.3 Loss

Default:

```text
weighted cross entropy + 0.5 * multiclass Dice loss
```

Calculate class weights from the training manifest, with clipping to avoid extreme seam weights.

### 11.4 Metrics

Report:

- per-class IoU,
- mean IoU,
- pixel accuracy,
- seam precision/recall/F1,
- confusion matrix.

Save RGB/mask/prediction overlays every epoch.

### 11.5 Training defaults

```text
optimizer: AdamW
learning rate: 1e-4
batch size: 8, configurable
epochs: 50
early stopping: validation mean IoU, patience 8
gradient clipping: 1.0
mixed precision: optional
```

### Acceptance criteria

- Overfits a 16-sample fixture to mean IoU >= 0.98.
- On non-randomized validation, target mean IoU >= 0.95.
- Predicted masks can be passed directly to the VSN.

---

## 12. Segment 7 — Position Alignment Network

### Goal

Train the 21x21 XY correction heatmap.

### 12.1 Model

Input:

```text
[B, 1, 200, 250]
```

The mask must be encoded consistently. Default encoding:

```text
background=0.0, peg=0.5, seam=1.0
```

Output:

```text
[B, 2, 21, 21] logits
```

Channel 0 is background; channel 1 is target cell.

### 12.2 Loss

Use pixel-wise cross entropy on the 21x21 grid.

Because one positive cell is highly imbalanced, apply class weighting or focal loss. Required default:

```text
cross entropy with positive class weight derived from 440:1 imbalance,
clipped to a configurable maximum.
```

### 12.3 Decode

```python
prob = softmax(logits, dim=1)[:, 1]
flat_index = argmax(prob)
row, col = unravel(flat_index)
dx_m, dy_m = decode_position(row, col)
confidence = max(prob)
```

Do not create a full binary map and search for equality with the global maximum.

### 12.4 Metrics

- exact-cell accuracy,
- accuracy within 1/2/5 cells,
- mean absolute X error in mm,
- mean absolute Y error in mm,
- radial error in mm,
- confidence calibration.

Evaluate separately on:

1. ground-truth masks,
2. predicted segmentation masks,
3. seen shapes,
4. unseen shapes.

### 12.5 Training defaults

```text
optimizer: AdamW
learning rate: 1e-4
epochs: 50
batch size: 16
early stopping metric: validation radial error
```

### Acceptance criteria

- Overfits a tiny fixture.
- Coordinate sign tests pass.
- Ground-truth-mask validation radial error is <= 1.0 mm.

---

## 13. Segment 8 — Orientation Alignment Network

### Goal

Train the Siamese orientation matcher and output an 11-value orientation heatmap.

### 13.1 Inputs

From a segmentation mask:

```python
peg_mask = mask == 1
seam_mask = mask == 2
```

Rotate the seam mask around the exact crop center for each candidate angle.

Use one rotation implementation everywhere. Prefer a tensor operation with:

- nearest-neighbor interpolation,
- zero fill,
- no expansion,
- explicitly tested angle sign.

### 13.2 Model

Shared encoder:

```text
input:  [B, 1, 200, 250]
output: feature map or feature vector
```

The paper states a three-channel feature map; the existing code uses 64 channels. Make feature dimension configurable, default `32`, and record it in checkpoints.

Batch algorithm:

1. Encode peg masks once.
2. Encode all 11 rotated seam masks.
3. Calculate distances.
4. Convert distance to score:

```python
orientation_scores = exp(-distance / temperature)
orientation_probs = scores / scores.sum()
```

### 13.3 Loss

Required default:

```text
cross entropy over negative distances for the 11 candidates
+ optional margin-ranking auxiliary loss
```

This is more stable than the current single-positive plus mean-clamped-negative loop while preserving the paper’s matching objective.

Provide a `paper_margin` mode:

```text
max(D_negative - D_positive + margin, 0)
```

with `margin=1`.

### 13.4 Decode

```python
index = argmax(orientation_probs)
dyaw_deg = candidate_angles[index]
confidence = orientation_probs[index]
```

### 13.5 Metrics

- exact candidate accuracy,
- accuracy within 2 degrees,
- accuracy within 4 degrees,
- mean absolute angular error,
- confusion matrix,
- confidence calibration.

### Acceptance criteria

- Rotation sign test passes with synthetic asymmetric masks.
- Ground-truth-mask validation MAE is <= 2 degrees.
- Batch inference returns `[B, 11]`.

---

## 14. Segment 9 — Unified VSN

### Goal

Remove duplicated perception logic from scripts.

Create:

```python
class VirtualSensorNetwork(nn.Module):
    def forward(
        self,
        rgb: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> VSNOutput:
        ...
```

Exactly one of `rgb` or `mask` is required.

`VSNOutput`:

```python
{
    "mask_logits": Tensor | None,          # [B, 3, 200, 250]
    "mask": Tensor,                        # [B, 200, 250]
    "position_logits": Tensor,             # [B, 2, 21, 21]
    "position_prob": Tensor,               # [B, 21, 21]
    "orientation_scores": Tensor,          # [B, 11]
    "orientation_prob": Tensor,            # [B, 11]
    "dxy_m": Tensor,                       # [B, 2]
    "dyaw_deg": Tensor,                    # [B]
    "position_confidence": Tensor,          # [B]
    "orientation_confidence": Tensor,       # [B]
}
```

### Modes

```text
oracle-mask: environment ground-truth mask
predicted-mask: segmentation model prediction
```

The position and orientation models are frozen by default when training controllers. End-to-end fine-tuning may be an optional ablation.

### Acceptance criteria

- CPU and CUDA inference produce matching decoded outputs within tolerance.
- Batch size 1 and batch size N work.
- Loading three checkpoints produces one ready-to-use VSN.

---

## 15. Segment 10 — SFSS Controller and Evaluation

### Goal

Implement the supervised single-frame policy and recursive closed-loop baseline.

### 15.1 Controller

```python
class SFSSController:
    def reset(self) -> None: ...
    def act(self, vsn_output: VSNOutput) -> ControllerAction: ...
```

Default action:

```python
dx_cmd = -gain_xy * predicted_dx
dy_cmd = -gain_xy * predicted_dy
dyaw_cmd = -gain_yaw * predicted_dyaw
```

Default gains:

```text
gain_xy = 0.7
gain_yaw = 0.7
```

Clip commands to environment physical action limits and convert them to normalized actions.

### 15.2 Confidence handling

Provide modes:

```text
ignore
scale
hold
```

Default `scale`:

```python
action *= min(position_confidence, orientation_confidence) / threshold
```

clipped to `[0, 1]`.

### 15.3 Evaluation variants

Required:

1. Oracle pose controller.
2. VSN with ground-truth mask.
3. Full RGB pipeline with predicted mask.
4. One-step SFSS.
5. Recursive SFSS up to `max_steps`.

### 15.4 Episode artifacts

Optionally save:

- RGB frames,
- masks,
- heatmaps,
- action arrows,
- pose-error trajectory,
- GIF/MP4,
- per-step JSON.

### Acceptance criteria

- Oracle variant succeeds >= 99%.
- Ground-truth-mask SFSS runs on every shape without special cases.
- Evaluation writes aggregate and per-shape results.

---

## 16. Segment 11 — SFMS Reinforcement Learning

### Goal

Train a single-frame, multi-step controller using the current VSN heatmaps as state.

### 16.1 Observation

Canonical state:

```python
state = concatenate(
    flatten(position_prob),   # 441
    orientation_prob,         # 11
)                              # total 452
```

Optional appended values:

```text
position confidence
orientation confidence
normalized previous action
normalized step fraction
```

The required baseline uses exactly 452 values.

### 16.2 Action

Use continuous normalized `[dx, dy, dyaw]` action matching the environment.

### 16.3 Algorithm

Implement a clean Advantage Actor-Critic baseline because the paper uses A2C.

Required network:

```text
452 -> 256 -> 128
actor mean head: 3
learned log standard deviation: 3
critic head: 1
activation: Tanh
```

Use:

```text
gamma=0.99
GAE lambda=0.95
actor lr=3e-4
critic lr=1e-3
entropy coefficient=0.01
value coefficient=0.5
gradient norm clip=0.5
```

The implementation may reuse mathematical utilities, but must not depend on the current experimental `a2c*` modules.

### 16.4 Training curriculum

Use stages:

1. XY only, ground-truth mask, small errors.
2. XY+yaw, ground-truth mask, small errors.
3. Full ±10 mm/±10 degree range.
4. Predicted segmentation masks.
5. Medium domain randomization.
6. Optional occlusion curriculum.

Advance based on rolling success threshold, default 80%, or a maximum stage step count.

### 16.5 VSN behavior

Default:

- segmentation/position/orientation models in evaluation mode,
- no gradients,
- VSN outputs detached before controller input.

### 16.6 Vector environments

Implement a Gym-0.26-compatible synchronous or subprocess vector runner that:

- preserves terminated versus truncated,
- auto-resets only after recording terminal observations,
- carries final info,
- seeds each worker independently,
- closes all PyBullet clients.

### 16.7 Evaluation

Use deterministic actor mean, not sampled actions.

Report results separately for:

- train-seen,
- validation-unseen,
- test-unseen,
- ground-truth masks,
- predicted masks.

### Acceptance criteria

- Random-policy smoke test completes.
- A 1,000-update training smoke run has finite losses and gradients.
- SFMS beats random policy by a statistically meaningful margin.
- No ground-truth pose error is included in the policy observation.

---

## 17. Segment 12 — MFMS Recurrent Reinforcement Learning

### Goal

Use heatmap history to reduce single-frame ambiguity.

### 17.1 Observation sequence

At step `t`, supply:

```text
[state_(t-N+1), ..., state_t]
```

where each state is the 452-value SFMS vector.

Default history length:

```text
N = 4
```

Pad episode starts with zeros and provide a validity mask.

### 17.2 Network

Required baseline:

```text
per-frame projection: 452 -> 256 with ReLU
single-layer LSTM: input 256, hidden 256
actor head: hidden -> 3
critic head: hidden -> 1
```

Reset hidden state on episode termination or truncation.

### 17.3 Training

Use recurrent A2C with:

- sequence-preserving rollout batches,
- burn-in optional and disabled by default,
- masked losses on padded sequence entries,
- gradient clipping,
- deterministic hidden-state handling in evaluation.

Do not flatten shuffled timesteps before LSTM training.

### 17.4 Required robustness experiment

Add synthetic occlusion:

- hide 10–40% of crop,
- apply for one or more consecutive frames,
- ensure masks and RGB are occluded consistently when testing predicted segmentation.

Compare SFSS, SFMS, and MFMS on identical episode seeds.

### Acceptance criteria

- Hidden state resets correctly.
- Padded entries do not affect loss.
- MFMS evaluation runs with deterministic recurrent state.
- Report whether MFMS improves occlusion success over SFMS; improvement is a target, not a hard completion gate.

---

## 18. Segment 13 — Evaluation and Reporting

### 18.1 Core metrics

Per episode:

```text
success
steps
initial/final dx
initial/final dy
initial/final radial XY error
initial/final yaw error
cumulative reward
termination reason
insertion depth, if applicable
collision failure, if applicable
```

Aggregate:

- success rate with 95% confidence interval,
- mean/median steps among successful episodes,
- final error mean/std/percentiles,
- out-of-bounds rate,
- collision failure rate,
- inference time per module and end-to-end,
- results per shape and split.

### 18.2 Required experiment matrix

Run at least:

| Controller | Mask | Task |
|---|---|---|
| Oracle | N/A | alignment |
| SFSS one-step | ground truth | alignment |
| SFSS recursive | ground truth | alignment |
| SFSS recursive | predicted | alignment |
| SFMS | ground truth | alignment |
| SFMS | predicted | alignment |
| MFMS | ground truth | alignment |
| MFMS | predicted | alignment |
| Best controller | predicted | insertion |

### 18.3 Required ablations

1. Position-only versus position+yaw.
2. Ground-truth versus predicted masks.
3. No versus medium domain randomization.
4. SFSS versus SFMS versus MFMS.
5. History lengths `1, 2, 4, 8`.
6. Initial ranges:
   - ±5 mm/±5 degrees,
   - ±10 mm/±10 degrees.
7. Success tolerances:
   - 0.5 mm,
   - 0.6 mm,
   - 1.0 mm.
8. No occlusion versus increasing occlusion.

### 18.4 Repetition

Default final evaluation:

```text
100 episodes per shape per method
5 independent seeds for learned controller training when compute permits
```

Smoke evaluation may use 3–10 episodes.

### 18.5 Outputs

Generate:

- `summary.json`,
- `episodes.csv`,
- `per_shape.csv`,
- Markdown result tables,
- success-versus-step plots,
- error trajectory plots,
- confusion matrices,
- representative success/failure videos.

### Acceptance criteria

- Every reported number can be traced to episode records.
- Identical evaluation seeds give identical initial states across controllers.
- Failed episodes are retained, not silently dropped.

---

## 19. Segment 14 — Testing Strategy

### 19.1 Unit tests

Required tests:

1. Position encode/decode round trip.
2. Orientation encode/decode.
3. Coordinate signs.
4. Action normalization and clipping.
5. Reward improvement and regression cases.
6. Success boundaries exactly at tolerance.
7. Shape split disjointness.
8. Checkpoint save/load.
9. Rotation sign and center.
10. Mask class composition.

### 19.2 Environment contract tests

For each representative shape:

- `reset()` output belongs to observation space.
- `step()` accepts sampled action.
- step output uses five-value API.
- `terminated` and `truncated` are never both incorrectly asserted.
- maximum step truncation works.
- explicit pose reset is honored.
- same seed is reproducible.
- close is idempotent.

### 19.3 Model tests

- Expected tensor shapes.
- Forward/backward pass has finite gradients.
- Tiny dataset overfit.
- CPU checkpoint load.
- VSN accepts either RGB or mask.
- Invalid simultaneous/missing inputs raise clear errors.

### 19.4 Integration tests

1. Collect 32 samples.
2. Validate and load them.
3. Train each perception model for a few batches.
4. Build VSN from checkpoints.
5. Run five SFSS episodes.
6. Run a short RL rollout.
7. Generate a report.

### 19.5 Regression tests

Store a small fixed seed fixture containing:

- expected pose labels,
- mask checksums or tolerant pixel statistics,
- oracle convergence result,
- reward trajectory.

Avoid exact RGB checksums when renderer versions make them unstable.

---

## 20. Segment 15 — Command-Line Contracts

All entry points must support `--config`, repeatable key overrides, and `--seed`.

Required commands:

```powershell
python scripts/validate_assets.py --config configs/base.yaml

python scripts/collect_dataset.py `
  --config configs/data.yaml `
  --split train_seen

python scripts/train_segmentation.py --config configs/segmentation.yaml
python scripts/train_position.py --config configs/position.yaml
python scripts/train_orientation.py --config configs/orientation.yaml

python scripts/evaluate.py `
  --config configs/sfss.yaml `
  --method sfss `
  --mask_source ground_truth

python scripts/train_sfms.py --config configs/sfms.yaml
python scripts/train_mfms.py --config configs/mfms.yaml

python scripts/evaluate.py `
  --config configs/evaluation.yaml `
  --method all `
  --mask_source predicted `
  --task insertion

python scripts/run_demo.py `
  --method sfss `
  --shape square-concave1 `
  --gui
```

Every command must have:

- `--help`,
- clear missing-checkpoint errors,
- nonzero exit status on failure,
- resolved configuration output before execution.

---

## 21. Configuration Defaults

`configs/base.yaml` should expose at least:

```yaml
project:
  seed: 1
  device: auto
  output_root: artifacts

environment:
  task: alignment
  max_steps: 20
  xy_initial_range_mm: 10.0
  yaw_initial_range_deg: 10.0
  xy_success_axis_mm: 1.0
  yaw_success_deg: 2.0
  xy_workspace_mm: 15.0
  yaw_workspace_deg: 15.0
  max_action_xy_mm: 2.0
  max_action_yaw_deg: 2.0
  reward_mode: dense_progress
  gui: false

camera:
  render_width: 1280
  render_height: 720
  crop_width: 250
  crop_height: 200
  fov_y_deg: 45.0

vsn:
  position_grid_size: 21
  position_resolution_mm: 1.0
  orientation_angles_deg: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
  mask_source: ground_truth

evaluation:
  episodes_per_shape: 100
  save_videos: false
```

Configuration validation must reject incompatible values, such as a position grid that cannot cover the initial range.

---

## 22. Implementation Order and Dependency Gates

Do not implement all components simultaneously. Use these gates.

### Milestone A — Stable simulator

Segments:

```text
0, 1, 2, 3
```

Gate:

- assets validate,
- environment tests pass,
- oracle alignment succeeds >=99%.

### Milestone B — Physical insertion simulation

Segment:

```text
4
```

Gate:

- exact-alignment insertion passes all shapes,
- intentional misalignment fails.

### Milestone C — Offline data and perception

Segments:

```text
5, 6, 7, 8, 9
```

Gate:

- dataset validates,
- all models overfit tiny fixtures,
- VSN runs end-to-end.

### Milestone D — Supervised controller

Segment:

```text
10
```

Gate:

- SFSS evaluation table exists for all splits.

### Milestone E — RL controllers

Segments:

```text
11, then 12
```

Gate:

- SFMS exceeds random policy,
- MFMS recurrent tests pass.

### Milestone F — Final benchmark

Segments:

```text
13, 14, 15
```

Gate:

- reproducible full report,
- all required tests pass,
- best predicted-mask controller completes simulated insertion.

---

## 23. Non-Negotiable Implementation Rules

1. Do not use `obs["pose_error"]` as controller input except for the oracle baseline.
2. Do not train and evaluate on random samples from the same shape when claiming unseen-shape generalization.
3. Do not use RGB color thresholding as the authoritative mask source.
4. Do not ignore actions during training.
5. Do not overload one mode flag to change the meaning of `step()`.
6. Do not serialize whole model objects.
7. Do not duplicate position/orientation decoding across scripts.
8. Do not silently clip invalid dataset labels without recording out-of-range counts.
9. Do not use Panda IK for the required simulation.
10. Do not mark alignment success as physical insertion success in insertion-mode reports.
11. Do not continue training after NaN/Inf loss; fail and save diagnostics.
12. Do not select final checkpoints using test-unseen results.

---

## 24. Final Completion Checklist

The simulation is complete only when all boxes below are satisfied:

### Environment

- [ ] All assets validated.
- [ ] Gym spaces match emitted data.
- [ ] Deterministic reset and step behavior.
- [ ] Reward, termination, and truncation implemented.
- [ ] Oracle alignment passes.
- [ ] Standalone-object geometric insertion passes.

### Data and perception

- [ ] Shape-disjoint datasets generated.
- [ ] Segmentation model trains and evaluates.
- [ ] Position model trains and evaluates.
- [ ] Orientation model trains and evaluates.
- [ ] Unified VSN supports ground-truth and predicted masks.

### Controllers

- [ ] SFSS one-step and recursive modes implemented.
- [ ] SFMS A2C implemented and trained.
- [ ] MFMS recurrent A2C implemented and trained.
- [ ] No learned policy consumes ground-truth pose.

### Evaluation

- [ ] Seen/unseen tables generated.
- [ ] Ground-truth/predicted-mask comparison generated.
- [ ] Robustness and occlusion tests generated.
- [ ] Alignment and insertion results are separately reported.
- [ ] Episode-level records and aggregate reports saved.

### Engineering quality

- [ ] Unit tests pass.
- [ ] Integration smoke test passes.
- [ ] Checkpoints are portable.
- [ ] Runs are reproducible from configuration and seed.
- [ ] No real robotic interface has been implemented.

---

## 25. Expected Final User Workflow

After implementation, a new user should be able to:

```text
1. Create the Python environment.
2. Validate assets.
3. Generate simulation datasets.
4. Train segmentation, position, and orientation models.
5. Evaluate SFSS.
6. Train and evaluate SFMS.
7. Train and evaluate MFMS.
8. Run a GUI demonstration.
9. Produce one report comparing all methods on seen and unseen shapes.
```

No source edits, hard-coded absolute paths, physical robot, or undocumented manual steps should be required.
