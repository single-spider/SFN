# Panda Arm Peg-in-Hole Simulation and Execution Validation Specification

**Date:** 2026-07-01  
**Status:** Implementation specification / handoff document  
**Primary purpose:** Build and validate a first-class Panda-arm execution layer for the SFN peg-in-hole project, separate from the current Cartesian-only ML/RL work.

---

## 1. Why this document exists

The current SFN project has a working clean simulated alignment stack:

```text
RGB or mask
-> VSN perception
-> SFSS / SFMS controller
-> direct Cartesian peg pose update
-> alignment metrics
```

That is useful for ML/controller development, but it does **not** prove that a Franka Panda arm can execute the same alignment/insertion motions.

This document defines the separate robotics workpiece:

```text
controller action [dx, dy, dyaw]
-> Panda Cartesian end-effector target
-> inverse kinematics / joint control
-> attached peg motion
-> measured pose validation
-> optional contact insertion validation
```

The goal is not to replace the current SFMS/MFMS work. The goal is to create a reusable robot execution layer so the learned controllers can later be run through a simulated Panda, and eventually mapped to a real robot workflow.

---

## 2. Required reader context

Before implementing, read these files:

```text
HANDOVER.md
SIMULATION_TECH_SPEC.md
peg-in-hole-sfn/artifacts/sfms_training_20260630/REPORT.md
peg-in-hole-sfn/gymEnv/envs/peg_in_hole_v11.py
peg-in-hole-sfn/demo_closed_loop_gui.py
peg-in-hole-sfn/demo_object_insertion_gui.py
peg-in-hole-sfn/view_pybullet_scene.py
peg-in-hole-sfn/sfn/envs/alignment_env.py
peg-in-hole-sfn/sfn/envs/insertion_env.py
peg-in-hole-sfn/sfn/models/controllers.py
peg-in-hole-sfn/sfn/evaluation/evaluate_sfms.py
```

Important existing checkpoints:

```text
peg-in-hole-sfn/models/sfms.pt
peg-in-hole-sfn/models/sfms_strict_xy_anchor1_best.pt
peg-in-hole-sfn/models/segmentation.pt
peg-in-hole-sfn/models/position.pt
peg-in-hole-sfn/models/orientation.pt
```

Important current configs:

```text
peg-in-hole-sfn/configs/sfms.yaml
peg-in-hole-sfn/configs/sfms_strict.yaml
peg-in-hole-sfn/configs/sfms_strict_xy.yaml
```

External references to consult during implementation:

```text
PyBullet official Quickstart Guide / API docs
PyBullet Panda examples in pybullet_data/franka_panda
Franka Emika Panda kinematic conventions and joint limits
Project paper: 2204.07776v2.pdf
```

---

## 3. Current state

### 3.1 What already works

The project currently has:

1. A clean direct Cartesian alignment environment under `peg-in-hole-sfn/sfn/envs/`.
2. A standalone-object insertion visual demo:

```text
peg-in-hole-sfn/demo_object_insertion_gui.py
```

3. A legacy Panda-based visual/demo environment:

```text
peg-in-hole-sfn/gymEnv/envs/peg_in_hole_v11.py
```

4. A GUI closed-loop Panda demo:

```text
peg-in-hole-sfn/demo_closed_loop_gui.py
```

5. Strong learned/teacher SFMS controllers for direct Cartesian alignment:

```text
models/sfms.pt
models/sfms_strict_xy_anchor1_best.pt
```

### 3.2 What does not yet work as a validated Panda simulation

The current Panda path is not yet a validated robot execution benchmark.

Known problems:

1. `peg_in_hole_v11.py` uses old Gym API:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

instead of Gym 0.26:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

2. Observation and action spaces are wrong/misleading:

```text
observation_space = Box(...)
action_space = Discrete(4)
```

but the actual observation is a dictionary and the action is `[dx, dy, dyaw]`.

3. In non-test mode, the environment ignores supplied actions and samples random pose errors.

4. `step()` always returns:

```text
reward = 0
done = False
info = {}
```

so it is not a proper RL or evaluation environment.

5. `dyaw` reported by the environment is an accumulator, not a measured end-effector/peg yaw.

6. `dxy` is measured from the Panda end-effector link state:

```python
dxy = [endPos[0] + 1, endPos[1]]
```

but there is no complete validation that this equals the actual peg tip / peg geometry pose.

7. The Panda IK command may not execute the requested Cartesian correction exactly.

8. The peg attachment transform to the end-effector is not formally documented or tested.

9. There is no measured command-tracking report:

```text
commanded dx/dy/dyaw
vs
actual peg dx/dy/dyaw
```

10. There is no robust contact insertion validation using the Panda arm.

11. Mask generation still uses fragile RGB thresholding and current-working-directory-relative paths in the legacy env.

12. Pyrender renderer lifecycle is not cleanly managed in the legacy env.

13. Collision/visual geometry consistency has not been validated for Panda insertion.

---

## 4. Scope

### 4.1 In scope

Implement and validate:

1. Panda simulation loading.
2. Peg attachment to Panda end-effector.
3. Hole/base loading.
4. Cartesian command interface:

```text
dx_m, dy_m, dyaw_deg
```

5. IK execution and joint control.
6. Measured peg pose extraction.
7. Command tracking tests.
8. Alignment environment using measured Panda/peg state.
9. Optional VSN/SFSS/SFMS controller integration.
10. Z descent and insertion validation.
11. Deterministic test and report artifacts.

### 4.2 Out of scope for this spec

Do **not** implement real hardware communication yet:

```text
no ROS/ROS2 nodes
no Franka Control Interface
no MoveIt deployment
no real robot commands
no real gripper commands
no physical safety system
```

However, the code should be designed so a real robot bridge can later implement the same command interface.

---

## 5. Design principle

The Panda layer must be treated as a robotics execution layer, not an ML training shortcut.

The learned controller should keep the same simple output:

```text
action = [dx, dy, dyaw]
```

The Panda layer is responsible for answering:

```text
Can the robot actually execute this action accurately?
```

Do not train SFMS/MFMS directly against an unvalidated Panda environment. First validate that the Panda can execute known oracle commands.

---

## 6. Target repository layout

Add a new package area under:

```text
peg-in-hole-sfn/sfn/panda/
```

Suggested files:

```text
peg-in-hole-sfn/sfn/panda/
  __init__.py
  config.py
  robot_model.py
  peg_attachment.py
  kinematics.py
  command.py
  measurement.py
  panda_scene.py
  panda_alignment_env.py
  panda_insertion_env.py
  validation.py
  reporting.py
```

Suggested scripts:

```text
peg-in-hole-sfn/scripts/panda_validate_model.py
peg-in-hole-sfn/scripts/panda_validate_ik.py
peg-in-hole-sfn/scripts/panda_validate_attachment.py
peg-in-hole-sfn/scripts/panda_validate_command_tracking.py
peg-in-hole-sfn/scripts/panda_evaluate_oracle.py
peg-in-hole-sfn/scripts/panda_evaluate_controller.py
peg-in-hole-sfn/scripts/panda_run_demo.py
```

Suggested tests:

```text
peg-in-hole-sfn/tests/test_panda_model.py
peg-in-hole-sfn/tests/test_panda_kinematics.py
peg-in-hole-sfn/tests/test_panda_attachment.py
peg-in-hole-sfn/tests/test_panda_command_tracking.py
peg-in-hole-sfn/tests/test_panda_alignment_env.py
peg-in-hole-sfn/tests/test_panda_insertion_env.py
```

Suggested artifacts folder:

```text
peg-in-hole-sfn/artifacts/panda_validation/
```

---

## 7. Coordinate and action contract

The Panda layer must preserve the existing project action convention.

### 7.1 Existing learned controller action

Controllers output normalized actions or physical actions.

Physical action:

```text
[dx_m, dy_m, dyaw_deg]
```

Meaning:

```text
incremental correction command
```

The current non-Panda direct environment applies it directly to the pose error.

### 7.2 Panda target command

For Panda execution, convert the physical command into an end-effector/peg target:

```text
current_target_pose
+ dx/dy in the documented hole/task frame
+ dyaw about task +Z
```

Important:

```text
Do not assume PyBullet world X/Y equals image/task X/Y without tests.
```

Create a transform object:

```python
TaskToWorldTransform
```

It must provide:

```python
task_delta_to_world_delta(dx_m, dy_m) -> np.ndarray[3]
task_yaw_to_world_rotation(dyaw_deg) -> quaternion or rotation matrix
```

### 7.3 Cardinal direction tests

The implementation must pass:

```text
command +1 mm task X -> measured peg task X increases by +1 mm within tolerance
command -1 mm task X -> measured peg task X decreases by -1 mm within tolerance
command +1 mm task Y -> measured peg task Y increases by +1 mm within tolerance
command -1 mm task Y -> measured peg task Y decreases by -1 mm within tolerance
command +1 deg yaw -> measured peg yaw increases by +1 deg within tolerance
command -1 deg yaw -> measured peg yaw decreases by -1 deg within tolerance
```

These tests are mandatory before any learned controller is evaluated on Panda.

---

## 8. Panda scene requirements

### 8.1 Robot model

Load a Panda robot in PyBullet.

Required checks:

1. Joint count is expected.
2. Revolute joint indices are documented.
3. Finger/gripper joints are documented.
4. End-effector link index is documented.
5. Joint limits are read from URDF.
6. Initial joint configuration is collision-free.
7. End-effector pose is finite and deterministic.

### 8.2 Peg attachment

The peg must be treated as rigidly attached to the Panda end-effector.

Two possible implementations:

1. Use existing shape asset `peg/peg.urdf` if it already includes Panda + peg.
2. Load Panda and standalone peg separately, then attach with a fixed constraint.

Preferred for validation:

```text
Load Panda and peg separately, attach with explicit fixed constraint.
```

Why:

```text
The peg-to-tool transform becomes explicit and testable.
```

Required data:

```python
PegAttachmentConfig(
    parent_link: int,
    child_body: int,
    parent_frame_pos: tuple[float, float, float],
    parent_frame_orn: tuple[float, float, float, float],
    child_frame_pos: tuple[float, float, float],
    child_frame_orn: tuple[float, float, float, float],
)
```

Required validation:

```text
After 1,000 simulation steps, peg relative transform to end-effector remains constant.
```

### 8.3 Hole/base model

Load the base/hole asset:

```text
peg-in-hole-sfn/gymEnv/envs/complex/<shape>/base/base.urdf
```

Required checks:

1. Base pose is fixed.
2. Base collision geometry exists.
3. Base visual geometry exists.
4. Hole opening center is known in world/task coordinates.
5. Shape asset does not silently fall back to a different shape.

---

## 9. IK and motion execution

### 9.1 IK function

Create:

```python
solve_ik(
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
    current_joint_positions: np.ndarray | None = None,
) -> np.ndarray
```

Use PyBullet IK initially.

Required controls:

```text
joint lower limits
joint upper limits
joint ranges
rest poses
max iterations
residual threshold
```

Do not call `calculateInverseKinematics()` with default unconstrained behavior and assume it is good.

### 9.2 Joint command execution

Create:

```python
execute_joint_target(
    joint_target: np.ndarray,
    steps: int,
    position_gain: float,
    velocity_gain: float,
    max_force: float,
) -> ExecutionResult
```

Return:

```python
ExecutionResult(
    commanded_ee_pose,
    measured_ee_pose,
    measured_peg_pose,
    joint_target,
    joint_actual,
    pos_error_m,
    yaw_error_deg,
    max_joint_error,
    contacts,
)
```

### 9.3 Cartesian command execution

Create:

```python
execute_cartesian_delta(dx_m: float, dy_m: float, dyaw_deg: float) -> ExecutionResult
```

This is the core bridge from SFN controller output to Panda motion.

### 9.4 Acceptance thresholds

For a single commanded small motion:

```text
translation tracking error <= 0.15 mm preferred, <= 0.30 mm acceptable
yaw tracking error <= 0.15 deg preferred, <= 0.30 deg acceptable
no unexpected contact during alignment motion
no joint limit violation
```

For accumulated closed-loop alignment:

```text
final XY <= 0.6 mm
final yaw <= 1.0 deg
```

---

## 10. Measurement requirements

The Panda layer must report actual measured pose, not just commanded pose.

Measure:

```text
end-effector pose
peg base pose
peg tip pose / insertion reference pose
hole center pose
pose error in task frame
```

Required object:

```python
MeasuredPandaState(
    joint_positions,
    joint_velocities,
    ee_pos_world,
    ee_quat_world,
    peg_pos_world,
    peg_quat_world,
    peg_tip_pos_world,
    peg_tip_quat_world,
    hole_pos_world,
    hole_quat_world,
    pose_error_task,
)
```

The reported error must be:

```text
pose_error = peg_reference_pose - hole_reference_pose
```

in the task frame.

Do not use controller accumulators as ground truth.

---

## 11. Validation stages

### Stage A — Load and static scene validation

Goal:

```text
Panda, peg, and base load deterministically.
```

Tests:

1. Load every shape.
2. Verify Panda joint metadata.
3. Verify peg and base body IDs.
4. Verify finite transforms.
5. Verify no initial self-collision or object collision.
6. Verify deterministic initial state with seed.

Artifacts:

```text
artifacts/panda_validation/stage_a_model_validation/summary.json
artifacts/panda_validation/stage_a_model_validation/per_shape.csv
```

Gate:

```text
All selected shapes load and close without PyBullet connection leaks.
```

### Stage B — Peg attachment validation

Goal:

```text
Peg remains rigidly attached to the tool.
```

Tests:

1. Move joints through safe poses.
2. Measure peg-to-end-effector transform.
3. Verify transform does not drift.
4. Verify peg visual and collision body move together.

Gate:

```text
relative transform drift <= 0.05 mm and <= 0.05 deg over 1,000 steps
```

### Stage C — IK reachability validation

Goal:

```text
Panda can reach the required alignment workspace around the hole.
```

Test grid:

```text
dx: [-10, -5, 0, 5, 10] mm
dy: [-10, -5, 0, 5, 10] mm
yaw: [-10, -5, 0, 5, 10] deg
```

For each target:

```text
solve IK
execute joint target
measure actual peg pose
record error
```

Gate:

```text
>= 99% targets reachable
mean translation tracking error <= 0.30 mm
mean yaw tracking error <= 0.30 deg
no target exceeds 1.0 mm or 1.0 deg unless listed as unreachable with reason
```

### Stage D — Incremental command tracking

Goal:

```text
The SFN action interface works through Panda.
```

Commands:

```text
[+1 mm, 0, 0]
[-1 mm, 0, 0]
[0, +1 mm, 0]
[0, -1 mm, 0]
[0, 0, +1 deg]
[0, 0, -1 deg]
combined random actions within ±2 mm / ±2 deg
```

Gate:

```text
command sign is correct for all cardinal directions
mean command tracking error <= 0.30 mm / 0.30 deg
```

### Stage E — Oracle Panda alignment

Goal:

```text
Known ground-truth correction succeeds through Panda execution.
```

Controller:

```text
oracle action = negative measured pose error, clipped to max action limits
```

Metrics:

```text
success rate
steps
final XY error
final yaw error
command tracking error per step
IK failure rate
contact during alignment
```

Gate:

```text
>= 95% success on all selected shapes
>= 99% success on synthetic-square or one representative simple shape
```

If oracle cannot succeed, do **not** evaluate SFSS/SFMS yet.

### Stage F — SFSS/SFMS Panda alignment

Goal:

```text
Run existing learned controllers through Panda execution.
```

Controllers:

```text
SFSS recursive predicted mask
SFMS teacher models/sfms.pt
SFMS strict-XY models/sfms_strict_xy_anchor1_best.pt
```

Start with ground-truth/known pose measurement for validation, then add VSN rendering if the Panda camera/render path is stable.

Gate:

```text
learned controller should approach direct Cartesian environment performance,
or any gap must be explained by measured Panda tracking errors.
```

### Stage G — Panda insertion validation

Goal:

```text
After alignment, Panda descends in Z and inserts the peg.
```

Procedure:

1. Align above hole.
2. Stop lateral/yaw motion.
3. Descend in small increments:

```text
0.1 mm to 0.25 mm per step
```

4. Monitor:

```text
contact points
normal forces / impulses if available
penetration depth
peg lateral drift
joint tracking error
```

5. Terminate:

```text
success: target insertion depth reached
failure: excessive contact/penetration, lateral drift, IK failure, timeout
```

Gate:

```text
exact alignment inserts successfully
intentional > tolerance misalignment fails
oracle-aligned insertion succeeds at high rate
```

---

## 12. Panda environment API

Create:

```python
class PandaPegInHoleAlignmentEnv(gym.Env):
    ...
```

Gym 0.26 API:

```python
obs, info = env.reset(seed=seed, options=options)
obs, reward, terminated, truncated, info = env.step(action)
```

Action:

```python
spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

Observation:

```python
{
    "rgb": uint8[3, 200, 250],
    "mask": uint8[200, 250],
    "pose_error": float32[3],
    "joint_positions": float32[n],
    "ee_pose": float32[7],
    "peg_pose": float32[7],
}
```

For learned policies, wrappers must hide:

```text
pose_error
joint_positions
ee_pose
peg_pose
```

Policy input must remain VSN state only unless explicitly testing a robot-state policy.

Info dictionary must include:

```python
{
    "shape": str,
    "seed": int,
    "step": int,
    "pose_error": np.ndarray[3],
    "xy_error_mm": float,
    "yaw_error_deg": float,
    "success": bool,
    "ik_success": bool,
    "tracking_error_mm": float,
    "tracking_yaw_error_deg": float,
    "contact_count": int,
    "unexpected_contact": bool,
    "action_normalized": np.ndarray[3],
    "action_physical": np.ndarray[3],
    "commanded_peg_pose": np.ndarray,
    "measured_peg_pose": np.ndarray,
}
```

---

## 13. Rendering and VSN integration

The Panda layer should initially validate robot motion without relying on VSN.

Recommended order:

1. Validate Panda motion using measured pose only.
2. Validate oracle alignment using measured pose.
3. Validate RGB/mask rendering from Panda scene.
4. Run VSN on rendered scene.
5. Run SFSS/SFMS from VSN output.

Do not debug IK, camera, segmentation, and RL simultaneously.

Rendering requirements:

```text
camera pose matches current SFN synthetic/direct camera as closely as possible
mask class values remain {0, 1, 2}
peg pixels come from segmentation/body ID where possible, not green threshold
hole mask aligns with base/hole opening
```

If exact legacy RGB style differs, that is acceptable, but report the difference.

---

## 14. Failure diagnosis checklist

When Panda alignment fails, classify failure into exactly one or more categories:

```text
IK_UNREACHABLE
IK_WRONG_BRANCH
JOINT_LIMIT
TRACKING_ERROR
ATTACHMENT_DRIFT
TASK_WORLD_SIGN_ERROR
YAW_SIGN_ERROR
PEG_REFERENCE_POINT_ERROR
CONTACT_DURING_ALIGNMENT
COLLISION_GEOMETRY_MISMATCH
CAMERA_RENDER_MISMATCH
VSN_PREDICTION_ERROR
CONTROLLER_ACTION_ERROR
```

Every failed episode must retain:

```text
episode JSON
initial pose
per-step commanded action
per-step measured pose
per-step tracking error
contacts
final screenshot if GUI/rendering enabled
```

---

## 15. Reports and artifacts

Every Panda validation script must write:

```text
summary.json
episodes.csv or trials.csv
per_shape.csv where applicable
config_resolved.json
README.md or REPORT.md
```

Recommended artifact structure:

```text
artifacts/panda_validation/
  model_validation_YYYYMMDD/
  attachment_validation_YYYYMMDD/
  ik_grid_YYYYMMDD/
  command_tracking_YYYYMMDD/
  oracle_alignment_YYYYMMDD/
  sfms_alignment_YYYYMMDD/
  insertion_YYYYMMDD/
```

---

## 16. Minimum command contracts

### Validate model loading

```powershell
..\.venv\Scripts\python.exe scripts\panda_validate_model.py `
  --shapes square-concave1,square-fillet4 `
  --out artifacts\panda_validation\model_validation_smoke
```

### Validate attachment

```powershell
..\.venv\Scripts\python.exe scripts\panda_validate_attachment.py `
  --shape square-concave1 `
  --steps 1000 `
  --out artifacts\panda_validation\attachment_smoke
```

### Validate IK grid

```powershell
..\.venv\Scripts\python.exe scripts\panda_validate_ik.py `
  --shape square-concave1 `
  --grid-mm -10,-5,0,5,10 `
  --grid-yaw-deg -10,-5,0,5,10 `
  --out artifacts\panda_validation\ik_grid_smoke
```

### Validate action tracking

```powershell
..\.venv\Scripts\python.exe scripts\panda_validate_command_tracking.py `
  --shape square-concave1 `
  --trials 100 `
  --out artifacts\panda_validation\command_tracking_smoke
```

### Evaluate oracle alignment

```powershell
..\.venv\Scripts\python.exe scripts\panda_evaluate_oracle.py `
  --split test_unseen `
  --episodes 20 `
  --out artifacts\panda_validation\oracle_alignment_test_unseen
```

### Evaluate SFMS through Panda

```powershell
..\.venv\Scripts\python.exe scripts\panda_evaluate_controller.py `
  --method sfms `
  --policy models\sfms_strict_xy_anchor1_best.pt `
  --segmentation models\segmentation.pt `
  --position models\position.pt `
  --orientation models\orientation.pt `
  --split test_unseen `
  --episodes 20 `
  --out artifacts\panda_validation\sfms_strict_xy_test_unseen
```

---

## 17. Completion checklist

The Panda workpiece is not complete until:

### Static model

- [ ] Panda loads deterministically.
- [ ] Base/hole loads deterministically.
- [ ] Peg loads and attaches deterministically.
- [ ] All body/link/joint indices are documented.
- [ ] No PyBullet connection leaks.

### Kinematics

- [ ] IK respects joint limits.
- [ ] IK reaches the full required workspace.
- [ ] End-effector tracking error is measured.
- [ ] Peg tracking error is measured.

### Action interface

- [ ] Cardinal direction signs pass.
- [ ] Yaw sign passes.
- [ ] Normalized action conversion matches existing SFN convention.
- [ ] Command tracking report exists.

### Alignment

- [ ] Oracle Panda alignment succeeds.
- [ ] SFSS/SFMS Panda alignment runs.
- [ ] Any gap from direct Cartesian alignment is explained.

### Insertion

- [ ] Exact alignment insertion succeeds.
- [ ] Intentional misalignment insertion fails.
- [ ] Contact/failure reasons are reported.
- [ ] Best controller insertion report exists.

### Documentation

- [ ] All known Panda caveats are documented.
- [ ] Reports are reproducible from commands.
- [ ] Real robot bridge assumptions are listed but not implemented.

---

## 18. Recommended implementation order

Do not start with learned controllers.

Recommended order:

```text
1. Load Panda + peg + base.
2. Explicitly attach peg to end-effector.
3. Measure peg reference pose.
4. Validate cardinal dx/dy/dyaw commands.
5. Validate IK grid.
6. Run oracle alignment through Panda.
7. Add rendering/VSN.
8. Run SFMS through Panda.
9. Add Z insertion/contact.
10. Produce final Panda validation report.
```

If any early stage fails, stop and debug that layer before continuing.

---

## 19. Key implementation warning

The current SFMS policy is not the risky part anymore.

The risky parts are:

```text
coordinate transforms
IK branch behavior
peg attachment transform
measured peg reference point
contact geometry
camera/render mismatch
```

Therefore, the Panda implementation should be judged first by mechanical execution accuracy, not by ML success rate.

Once the Panda can accurately execute:

```text
dx/dy/dyaw corrections
```

then SFSS/SFMS/MFMS can be plugged into it cleanly.

