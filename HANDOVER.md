# SFN Peg-in-Hole Handover Notes

## 2026-07-13 Software Completion Snapshot

The authoritative completion material is under `peg-in-hole-sfn/artifacts/software_completion_20260713`.

- Mesh-faithful Cartesian clean insertion: oracle, recursive SFSS, selected SFMS teacher and selected MFMS teacher each passed 40/40 held-out trials with sub-millimetre final XY error.
- Stabilized SFMS A2C did not improve the teacher (37/40 versus 40/40); the teacher remains selected.
- Paired robustness severity and consecutive burst-occlusion/history ablations are complete. Longer MFMS history reduced success and must not be presented as a temporal advantage.
- Standalone PyBullet physical insertion now passes exact alignment on 16/16 shapes and rejects a 2 mm offset on 16/16. The fixture collision was converted to a raster compound to eliminate unstable concave-facet contacts.
- Panda dynamic execution, corrected downward attachment, full-workspace IK and physical insertion are implemented. Exact insertion passes all 16 shapes.
- The selected Panda top-down camera was validated over 25,920 renders. A high-contrast blue simulated peg was introduced because a white peg was visually confounded with the Panda hand.
- The high-contrast Panda-native segmentation model reaches 0.9989 held-out mean IoU. Predicted-mask template pose reaches 0.261 mm mean XY and 1.621Â° mean yaw.
- Final held-out Panda dynamic insertion, 20 trials per cell: semantic masks SFSS/SFMS/MFMS = 20/20, 20/20, 19/20; predicted masks = 18/20, 20/20, 19/20. The blue-peg result is simulation-domain evidence, not sim-to-real evidence.
- The final report, consolidated matrix, raw episode records, Wilson intervals and plots are stored in the completion artifact directory.
- Hardware-only work remains: real camera and handâ€“eye calibration, real-image segmentation adaptation, part metrology, force/safety tuning and repeated physical trials.

## 2026-06-30 Current Roadmap Snapshot: Done vs Left

This section is intentionally placed near the top so a new session can quickly understand where the work stands.

### Big-picture plan

The project roadmap is:

```text
1. deterministic / oracle baseline
2. single-step learned/perception solver
3. multi-step RL solver
4. multi-step RL solver with history/recurrent memory
```

In paper terms, this maps roughly to:

```text
oracle / analytic baseline
SFSS: single-frame single-step
SFMS: single-frame multi-step RL
MFMS: multi-frame multi-step RL with history
```

### Current status table

| Method | Learns? | Uses history? | Current repo status |
|---|---:|---:|---|
| Oracle / deterministic | no | no | implemented; needs final locked report numbers |
| SFSS one-step | perception only | no | implemented; needs final eval/tuning |
| SFSS recursive | perception only | no recurrent memory | implemented; needs final eval/tuning |
| SFMS RL | yes | no | smoke implementation exists; needs real training |
| MFMS RL/history | yes | yes | not implemented yet; placeholder only |
| PyBullet/Panda visual smoke | no | no | direct insertion smoke passed; not integrated as main training env |

### Done / mostly done

#### 1. Deterministic/oracle solver

Status: mostly done as a baseline.

This is the clean â€œgod modeâ€ controller that uses the true simulated pose/error to compute corrective motion. It is useful for sanity checks, upper-bound/reference baselines, and debugging environment/action conventions.

Remaining need: run and record final report-quality oracle numbers for alignment and insertion.

#### 2. Perception stack

Status: mostly done.

Implemented/trained/evaluated pieces: segmentation, position estimation, orientation estimation, and VirtualSensorNetwork-style output.

Best known artifact-level metrics from previous evaluation:

```text
segmentation mean IoU: about 0.99995
position radial error: 0.0 mm in the strong artifact eval
orientation mean absolute error: about 0.58 deg
```

Caveat: some later `models/position.summary.json` output came from an experimental/scheduled run and looks worse than the earlier artifact evaluation. Treat `artifacts/perception_test` and fresh evals as stronger evidence than aborted/experimental monitor runs.

#### 3. Single-step solver / SFSS

Status: implemented and smoke-tested, but not yet final-performing.

Implemented: `SFSSController`, one-step mode, recursive mode, evaluation script support, per-step/per-episode artifacts, and optional visual panels.

Earlier smoke result with non-production checkpoints:

```text
sfss_gt_onestep_smoke:
  success_rate = 0.0
  mean_final_xy_error_mm = 1.68
  mean_final_yaw_error_deg = 4.62
```

That did not prove final failure; it only showed smoke checkpoints/eval thresholds were not yet report-ready.

#### 4. Multi-step non-history RL / SFMS

Status: code scaffold exists, but real training is not done.

Implemented: `sfn/training/train_sfms.py`, `SFMSActorCritic`, 452-dim VSN state, continuous 3D action, A2C-style smoke training, `sfn/evaluation/evaluate_sfms.py`, and policy checkpoint save/load/eval.

Earlier smoke result:

```text
sfms_smoke_eval:
  success_rate = 0.0
  mean_steps = 18
  mean_final_xy_error_mm = 21.1
  mean_final_yaw_error_deg = 9.11
```

So SFMS currently means â€œruns end-to-end as a smoke path,â€ not â€œtrained solved policy.â€

#### 5. PyBullet/Panda visual/sim smoke

Status: direct object-level smoke passed. This proves assets and direct insertion visualization work, but it is not yet the main training/evaluation environment and is not the full Panda arm controller.

### Not done / left

#### A. Finish and lock deterministic baseline numbers

Need final report-quality evaluation for oracle alignment/insertion, chosen final split/seed, fixed thresholds, episode CSV, summary JSON, and markdown report.

#### B. Cleanly evaluate SFSS

Need final SFSS evaluation for one-step and recursive modes, ground-truth and predicted-mask modes, same shape split/seed/thresholds as oracle, plus summary tables and artifacts.

#### C. Properly train SFMS RL

Major unfinished item: decide final state representation, train longer than smoke runs, tune reward/action scaling, evaluate on shape splits, and compare against oracle/SFSS.

#### D. Implement MFMS / history RL

Status: not implemented. `sfn/training/train_mfms.py` and `scripts/train_mfms.py` are placeholders.

Likely design: sequence/history input, GRU/LSTM actor-critic, dx/dy/dyaw output, A2C/PPO-style training, and the same metrics as SFMS.

### Immediate active work from 2026-06-30

The active task is to complete A and B above:

```text
A. run deterministic/oracle baseline evaluations and lock numbers
B. run SFSS one-step/recursive evaluations with ground-truth and predicted masks
then create a concise markdown report with commands, artifacts, and result tables
```

---
This document captures the current project context, what is in the repository, what has been restored so far, and a practical roadmap for completing a fuller simulation/training implementation of the paper.

## Project Context

Paper: `2204.07776v2.pdf`

Title: `Learning to Fill the Seam by Vision: Sub-millimeter Peg-in-hole on Unseen Shapes in Real World`

The core idea is to solve peg-in-hole alignment by looking at the seam/gap between the peg and hole. The system decomposes the task into:

1. Segmentation module:
   RGB image -> mask of background / peg / seam-hole region.

2. Position alignment module:
   segmentation mask -> 21x21 XY correction heatmap.

3. Orientation alignment module:
   peg mask + rotated seam masks -> yaw correction.

4. Controller / RL policy:
   uses the position/orientation outputs to command robot motion.

The useful mental model:

```text
camera image
-> segmentation mask
-> estimate dx, dy, dyaw
-> move robot
-> look again
-> repeat
```

The released repo appears strongest on the visual alignment side. The full physical insertion/contact part and exact paper RL wiring are not fully present as a clean runnable pipeline.

## Current Local Environment

A local virtual environment was created:

```text
.venv/
```

It is ignored by `.gitignore`.

Installed packages include:

```text
gym==0.26.2
pybullet
pyrender
trimesh
numpy
scipy
opencv-python
matplotlib
pillow
pytest
torch
```

Python used:

```text
Python 3.11.9
```

Note: Gym prints a warning about being unmaintained and NumPy 2.x. The current scripts still run despite this warning.

## Important Repository Structure

Top-level:

```text
README.md
2204.07776v2.pdf
content.md
HANDOVER.md
peg-in-hole-sfn/
assets/
```

`content.md` is rough human-style personal notes explaining the project in simpler language.

`HANDOVER.md` is this structured handover document.

Main code folder:

```text
peg-in-hole-sfn/
```

Important subfolders:

```text
peg-in-hole-sfn/gymEnv/
peg-in-hole-sfn/gymEnv/envs/
peg-in-hole-sfn/gymEnv/envs/complex/
peg-in-hole-sfn/algos/pytorch/fcn/
peg-in-hole-sfn/algos/pytorch/a2c/
peg-in-hole-sfn/algos/pytorch/a2c_rnn/
peg-in-hole-sfn/algos/pytorch/a2c_fusion/
peg-in-hole-sfn/utils/
```

## Simulation Environment

Main env file:

```text
peg-in-hole-sfn/gymEnv/envs/peg_in_hole_v11.py
```

Gym registration:

```text
peg-in-hole-sfn/gymEnv/__init__.py
```

Registered env id:

```text
gymEnv:peg-in-hole-v11
```

The environment returns observations like:

```python
obs = {
    "img": RGB image,             # shape roughly (3, 200, 250)
    "gt": segmentation mask,      # shape roughly (200, 250)
    "dxy": [dx, dy],              # current XY error
    "dyaw": yaw_error             # current yaw error
}
```

Important note:

`peg_in_hole_v11.step()` is primarily an alignment step. It changes XY/yaw. It does not implement a full physical insertion success condition. It returns:

```python
return obs, 0, False, {}
```

So it is not currently a complete RL environment.

### Patch Applied

Newer PyBullet returns camera pixels in a format that needed reshaping. This was patched in:

```text
peg-in-hole-sfn/gymEnv/envs/peg_in_hole_v11.py
```

Patch:

```python
rgb = np.reshape(rgb, (self.img_height, self.img_width, 4))
```

This allows `env.reset()` and `env.step()` to render correctly.

Also added:

```python
gui_mode=False
```

to the env constructor, so the env can run in PyBullet GUI mode when needed.

## PyBullet / Mesh Assets

The simulation assets exist in:

```text
peg-in-hole-sfn/gymEnv/envs/complex/
```

Available shapes include:

```text
square-concave1
square-concave2
square-convex1
square-convex2
square-convex3
square-convex4
square-diamond
square-fillet1
square-fillet2
square-fillet3
square-fillet4
square-hexagon
square-pentagon
square-square
square-trapezoid
square-triangle
```

Each shape usually contains:

```text
base/base.urdf
base/base.obj
peg/peg.urdf
peg/peg_test.urdf
peg/peg.obj
mask.obj
```

`peg.urdf` is attached to a Panda robot model.

`peg_test.urdf` is a standalone peg object, useful for clean visual demos.

## Main Existing Training/Inference Files

### Segmentation

```text
peg-in-hole-sfn/algos/pytorch/fcn/seg_ur5_real.py
peg-in-hole-sfn/algos/pytorch/fcn/unet.py
```

Purpose:

```text
RGB image -> 3-class segmentation mask
```

Network:

```python
UNet(3, 3)
```

Classes are roughly:

```text
0 background
1 peg
2 seam / visible hole region
```

Issue:

This imports `UR5Dataset` from `dataloader`, but `dataloader.py` source is missing in this checkout. Real-world segmentation training is therefore incomplete unless that dataloader is restored or rewritten.

### Position Alignment

```text
peg-in-hole-sfn/algos/pytorch/fcn/position_11.py
peg-in-hole-sfn/algos/pytorch/fcn/unet_11.py
peg-in-hole-sfn/utils/utils.py
peg-in-hole-sfn/train_position_11.py
```

Purpose:

```text
segmentation mask -> 21x21 XY heatmap
```

Network:

```python
UNet(1, 2)
```

The 21x21 heatmap represents possible XY corrections from:

```text
-10 mm to +10 mm in x
-10 mm to +10 mm in y
```

with 1 mm resolution.

The center cell is zero correction.

The helper `get_position_gt()` converts simulator `dxy` into a one-hot target heatmap.

### Orientation Alignment

```text
peg-in-hole-sfn/algos/pytorch/fcn/pose_8.py
peg-in-hole-sfn/algos/pytorch/fcn/unet.py
peg-in-hole-sfn/train_pose_8.py
```

Purpose:

```text
segmentation mask -> yaw correction
```

It separates the mask into:

```text
peg mask
seam/hole mask
```

Then it rotates the seam mask through candidate yaw values:

```text
-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 degrees
```

It passes peg mask and rotated seam masks through a shared U-Net and compares feature distances. The smallest distance gives the predicted yaw correction.

### Inference / Test Scripts

```text
peg-in-hole-sfn/test_pose_position_gui_11.py
peg-in-hole-sfn/test.py
peg-in-hole-sfn/test_pose_gui_8.py
```

`test_pose_position_gui_11.py` is the clearest existing closed-loop test script. It loads a position model and pose model, estimates `dx`, `dy`, and `dyaw`, then calls `env.step()`.

Important caveat:

These scripts mostly use `obs["gt"]` directly as the mask. The learned segmentation model is commented out.

### RL Code

Folders:

```text
peg-in-hole-sfn/algos/pytorch/a2c/
peg-in-hole-sfn/algos/pytorch/a2c_9/
peg-in-hole-sfn/algos/pytorch/a2c_rnn/
peg-in-hole-sfn/algos/pytorch/a2c_fusion/
peg-in-hole-sfn/algos/pytorch/a2c_rnn_encoder/
peg-in-hole-sfn/algos/pytorch/ppo/
peg-in-hole-sfn/algos/pytorch/sac/
peg-in-hole-sfn/algos/pytorch/td3/
peg-in-hole-sfn/algos/pytorch/ddpg/
```

The paper describes:

```text
SFSS: single-frame single-step
SFMS: single-frame multi-step with RL
MFMS: multi-frame multi-step with RNN/LSTM + RL
```

The repo has generic A2C/RNN/fusion RL implementations, but the exact paper-level training script that wires position/orientation heatmaps into RL was not found in clean source form.

Also, `peg_in_hole_v11` currently lacks reward/done logic, so a proper RL wrapper needs to be built.

## New Scripts Added

### `collect_sim_samples.py`

Path:

```text
peg-in-hole-sfn/collect_sim_samples.py
```

Purpose:

Collects reusable simulated samples from `gymEnv:peg-in-hole-v11`.
It saves compressed `.npz` chunks containing:

```text
rgb          uint8  [N, 3, 200, 250]
mask         uint8  [N, 200, 250]
dxy_m        float32 [N, 2]
dyaw_deg     float32 [N]
position_gt  float32 [N, 21, 21]
```

It also writes `metadata.json` and optional RGB/mask/overlay preview PNGs.

Run:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python collect_sim_samples.py --peg_types square-concave1 --samples_per_shape 100 --preview_count 5
```

Use all available shapes by omitting `--peg_types`:

```powershell
..\.venv\Scripts\python collect_sim_samples.py --samples_per_shape 1000 --output_dir data/sim_samples
```

Smoke test performed:

```powershell
..\.venv\Scripts\python collect_sim_samples.py --peg_types square-concave1 --samples_per_shape 2 --preview_count 1 --output_dir C:\tmp\sfn_sim_samples_smoke --overwrite
```

Verified output shapes:

```text
rgb: (2, 3, 200, 250)
mask: (2, 200, 250)
dxy_m: (2, 2)
dyaw_deg: (2,)
position_gt: (2, 21, 21)
```

### `simple_closed_loop_sim.py`

Path:

```text
peg-in-hole-sfn/simple_closed_loop_sim.py
```

Purpose:

Runs a simple closed-loop alignment simulation using simulator-provided `obs["dxy"]` and `obs["dyaw"]`.

This is an oracle controller, not the learned SFN model.

Run:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python simple_closed_loop_sim.py --episodes 3 --max_steps 10 --peg_type square-concave1 --seed 1
```

Verified result:

```text
3/3 successful episodes
```

Typical convergence:

```text
under 1 mm XY error and under 2 deg yaw error in about 2 correction steps
```

### `view_pybullet_scene.py`

Path:

```text
peg-in-hole-sfn/view_pybullet_scene.py
```

Purpose:

Opens PyBullet GUI and holds a static peg/hole scene.

Run:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python view_pybullet_scene.py --seconds 300 --peg_type square-concave1 --dx_mm 5 --dy_mm -4 --dyaw_deg 6
```

This is only a viewer, not an insertion demo.

### `demo_closed_loop_gui.py`

Path:

```text
peg-in-hole-sfn/demo_closed_loop_gui.py
```

Purpose:

Attempts to show Panda-based GUI alignment and z push.

Important caveat:

This is not a reliable insertion demo. The Panda IK/collision setup drifts and the `v11` env is mainly for alignment, not contact insertion. The final visual state may still show seam or look outside the hole.

This script is useful mainly to demonstrate why the current env is not a complete physical insertion simulator.

### `demo_object_insertion_gui.py`

Path:

```text
peg-in-hole-sfn/demo_object_insertion_gui.py
```

Purpose:

Clean visual concept demo. It directly loads the standalone peg and base URDFs, animates the peg from misaligned pose to aligned pose, then moves it down into the hole.

This bypasses the Panda arm and is only a visual concept demo.

Run:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python demo_object_insertion_gui.py --peg_type square-concave1 --hold_seconds 60
```

This is currently the clearest way to see the insertion concept visually.

## What Currently Works

Works:

```text
local Python env
PyBullet import
Pyrender/Trimesh import
gymEnv:peg-in-hole-v11 reset/render/step
headless closed-loop oracle alignment
PyBullet GUI viewer
object-level insertion visual demo
```

Partly works:

```text
Panda-based GUI alignment
```

Not yet working / not yet implemented:

```text
training position/orientation models end-to-end in this environment
loading pretrained SFN checkpoints
full learned closed-loop controller
proper RL environment reward/done wrapper
SFMS/MFMS paper reproduction
real-world UR5 deployment
physical Panda contact insertion success simulation
```

## Main Technical Caveats

1. The released source appears incomplete for exact paper reproduction.

2. `dataloader.py` is missing, which affects real-world segmentation training.

3. Several scripts reference old env versions like `v8` or `v12`, but only `v11` exists as source.

4. The training loops in `position_11.py` and `pose_8.py` have significant training sections commented out. These may need restoration before real training.

5. The RL code exists, but the env does not provide reward/done, and a clean heatmap-to-RL wrapper is missing.

6. The current `v11` env is best interpreted as a vision alignment environment, not a full contact-rich insertion simulator.

## Roadmap To Full Simulation Training

### Phase 1: Stabilize Simulation

Status: mostly done.

Tasks:

```text
keep env reset/render/step working
save sample obs images and masks
verify all shape assets load
decide whether success means alignment or physical insertion
```

Deliverable:

```text
stable environment smoke tests
visual debug scripts
```

### Phase 2: Restore Supervised Position Training

Tasks:

```text
inspect position_11.py commented training loop
uncomment/fix training update
train UNet(1,2) from obs["gt"] to 21x21 heatmap
save checkpoints under models/
```

Deliverable:

```text
position_model.pt
evaluation script for dx/dy prediction
```

### Phase 3: Restore Supervised Orientation Training

Tasks:

```text
inspect pose_8.py commented contrastive training loop
uncomment/fix training update
train UNet(1,64) feature matcher
save checkpoint under models/
```

Deliverable:

```text
pose_model.pt
evaluation script for dyaw prediction
```

### Phase 4: Build SFSS Evaluation

Tasks:

```text
write clean evaluate_sfss.py
use obs["gt"] masks first
position model predicts dx/dy
orientation model predicts dyaw
env.step applies correction
repeat for max N steps
```

Metrics:

```text
success rate
average steps
final XY error
final yaw error
seen vs unseen shapes
```

Success threshold:

```text
abs(dx) < 1 mm
abs(dy) < 1 mm
abs(dyaw) < 2 deg
```

Deliverable:

```text
SFSS table across shapes
```

### Phase 5: Add Segmentation Training In Simulation

Use simulation first, not real UR5.

Tasks:

```text
generate RGB/mask pairs from env
train UNet(3,3)
evaluate mask IoU
replace obs["gt"] with predicted mask
compare perfect mask vs predicted mask
```

Deliverable:

```text
seg_model.pt
full RGB -> mask -> VSN -> action simulation pipeline
```

### Phase 6: Build RL Wrapper

Current env is not enough for RL.

Need a wrapper:

```text
state = flattened position heatmap + orientation score vector
action = dx/dy/dyaw command
reward = improvement in alignment error - step penalty + success bonus
done = success or max steps
```

Possible SFMS state:

```text
21x21 position heatmap = 441 values
orientation scores = 11 values
total = 452 values
```

Reward idea:

```python
old_error = xy_weight * xy_error + yaw_weight * abs(yaw_error)
new_error = xy_weight * new_xy_error + yaw_weight * abs(new_yaw_error)
reward = old_error - new_error - step_penalty
if success:
    reward += success_bonus
    done = True
```

Deliverable:

```text
PegInHoleRlWrapper
random policy smoke test
```

### Phase 7: Train SFMS With RL

Start simple.

Recommended:

```text
use A2C or PPO
start with discrete action space
train on seen shapes
evaluate on unseen shapes
compare to SFSS
```

Possible discrete actions:

```text
dx in {-1, 0, +1} mm
dy in {-1, 0, +1} mm
dyaw in {-2, 0, +2} deg
```

Deliverable:

```text
trained SFMS policy
success table vs SFSS
```

### Phase 8: Train MFMS / Recurrent Policy

Only after SFMS works.

State:

```text
last N heatmap vectors
```

Model:

```text
GRU/LSTM encoder + actor/critic heads
```

Repo has RNN A2C variants, but they likely need adaptation.

Deliverable:

```text
MFMS policy
comparison under difficult/occluded cases
```

### Phase 9: Decide On Physical Insertion Simulation

Three options:

Option A:

```text
keep success as alignment only
```

This is simplest and closest to the current repo.

Option B:

```text
create object-level insertion env
directly move standalone peg body
define success by depth and collision state
```

Recommended if a visual/physical insertion simulator is needed.

Option C:

```text
repair full Panda IK/contact insertion
add z force/control
add contact-based success
```

Hardest and probably unnecessary for first reproduction.

## Recommended Next Steps

1. Sample collector added:

```text
collect_sim_samples.py
```

It saves RGB, mask, dxy, dyaw, and 21x21 position heatmaps for multiple shapes.

2. Restore position training first.

3. Restore orientation training second.

4. Build `evaluate_sfss.py`.

5. Only after SFSS is measurable, start RL wrapper.

The best near-term target is:

```text
train position + orientation on simulated gt masks
run closed-loop SFSS on unseen shapes
produce a success-rate table
```

That would reproduce the central paper idea before tackling RL.

---

## 2026-06-17 Implementation Progress: Clean `sfn` Simulation Pipeline

A new clean implementation track has been started under:

```text
peg-in-hole-sfn/sfn/
```

This follows `SIMULATION_TECH_SPEC.md` and is separate from the legacy experimental scripts. Legacy files remain available, but new command-line entry points import from `sfn.*` only.

### Milestone A: Stable Simulator Foundation

Status: implemented as an initial dependency-light scaffold.

Added package structure:

```text
peg-in-hole-sfn/sfn/
  config.py
  constants.py
  geometry.py
  seeding.py
  envs/
  data/
  models/
  training/
  evaluation/
```

Added configs:

```text
peg-in-hole-sfn/configs/base.yaml
peg-in-hole-sfn/configs/data.yaml
peg-in-hole-sfn/configs/segmentation.yaml
peg-in-hole-sfn/configs/position.yaml
peg-in-hole-sfn/configs/orientation.yaml
peg-in-hole-sfn/configs/sfss.yaml
peg-in-hole-sfn/configs/sfms.yaml
peg-in-hole-sfn/configs/mfms.yaml
peg-in-hole-sfn/configs/evaluation.yaml
```

Added scripts:

```text
peg-in-hole-sfn/scripts/validate_assets.py
peg-in-hole-sfn/scripts/collect_dataset.py
peg-in-hole-sfn/scripts/evaluate.py
peg-in-hole-sfn/scripts/run_demo.py
peg-in-hole-sfn/scripts/train_segmentation.py
peg-in-hole-sfn/scripts/train_position.py
peg-in-hole-sfn/scripts/train_orientation.py
peg-in-hole-sfn/scripts/train_sfms.py
peg-in-hole-sfn/scripts/train_mfms.py
```

Implemented foundation pieces:

```text
validated dataclass config loading
fallback YAML parser when PyYAML is unavailable
seed helper for Python / NumPy / Torch
checkpoint schema helpers using state_dicts, not full model serialization
shape split constants
canonical position and orientation codecs
```

Canonical position mapping now lives in:

```text
peg-in-hole-sfn/sfn/geometry.py
```

The required mapping is implemented and tested:

```python
col = round(-dx_m * 1000) + 10
row = round( dy_m * 1000) + 10
```

### Asset Registry

Implemented:

```text
peg-in-hole-sfn/sfn/envs/asset_registry.py
```

Public API:

```python
AssetRegistry.list_shapes()
AssetRegistry.get(shape)
AssetRegistry.validate(shape)
AssetRegistry.validate_all()
```

It discovers all 16 current shape directories under:

```text
peg-in-hole-sfn/gymEnv/envs/complex/
```

Validation checks required paths and, when optional dependencies are installed, can also check mesh bounds and PyBullet URDF loading. Missing optional dependencies are warnings unless `--strict-dependencies` is passed.

Run:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python scripts\validate_assets.py --config configs\base.yaml
```

### Deterministic Alignment Environment

Implemented:

```text
peg-in-hole-sfn/sfn/envs/alignment_env.py
```

Class:

```python
PegInHoleAlignmentEnv
```

Properties:

```text
Gym 0.26-style reset/step API
continuous normalized action space [-1, 1]^3
incremental dx / dy / dyaw corrections
workspace clipping
dense progress reward
success / terminated / truncated logic
per-env NumPy RNG
no Panda IK
no ignored actions
```

Observation:

```python
{
    "rgb": uint8 [3, 200, 250],
    "mask": uint8 [200, 250],
    "pose_error": float32 [3],
}
```

A dependency-light deterministic synthetic renderer is currently used in:

```text
peg-in-hole-sfn/sfn/envs/renderer.py
```

It preserves mask classes `{0, 1, 2}` and tensor contracts, while avoiding reliance on PyBullet/Pyrender during early tests. A PyBullet segmentation-ID renderer can later replace it behind the same interface.

### Data and Model Scaffolds

Added initial dataset helpers:

```text
sfn/data/schema.py
sfn/data/splits.py
sfn/data/collect.py
sfn/data/dataset.py
sfn/data/validate.py
```

Added initial model/API scaffolds:

```text
sfn/models/segmentation.py
sfn/models/position.py
sfn/models/orientation.py
sfn/models/vsn.py
sfn/models/controllers.py
```

Implemented:

```text
VirtualSensorNetwork forward API
VSNOutput dataclass
SFSSController scaffold
OracleController baseline
```

### Milestone B: Standalone Insertion Simulation

Status: implemented as a deterministic standalone-object insertion environment.

Implemented:

```text
peg-in-hole-sfn/sfn/envs/insertion_env.py
```

Class:

```python
PegInHoleInsertionEnv(PegInHoleAlignmentEnv)
```

It implements:

```text
ALIGN -> DESCEND -> SUCCESS / FAILURE
```

Behavior:

```text
agent controls dx/dy/dyaw only
when alignment success is reached, peg descends deterministically in Z
insertion success requires target depth and tighter residual insertion tolerance
intentional residual misalignment fails insertion
no Panda IK
no robot arm
no ROS / hardware middleware
```

Insertion config is now part of the main config dataclasses and YAML files:

```yaml
insertion:
  descent_increment_mm: 0.25
  target_depth_mm: 8.0
  max_descent_attempts: 64
  collision_mode: geometric
  insertion_xy_axis_mm: 0.6
  insertion_yaw_deg: 1.0
  max_collision_penetration_mm: 0.0
```

Current `collision_mode: geometric` uses a deterministic synthetic penetration proxy. This is deliberately isolated so future PyBullet contact queries can replace the proxy without changing the environment API.

Added public method:

```python
PegInHoleInsertionEnv.attempt_insertion()
```

This lets tests verify exact-alignment insertion and intentional-misalignment failure directly.

### Evaluation Updates

Updated:

```text
peg-in-hole-sfn/sfn/evaluation/evaluator.py
peg-in-hole-sfn/scripts/evaluate.py
```

Now supported:

```powershell
..\.venv\Scripts\python scripts\evaluate.py --task alignment --episodes 1
..\.venv\Scripts\python scripts\evaluate.py --task insertion --episodes 1
```

The oracle evaluator now records:

```text
task
success
steps
reward
final XY error
final yaw error
termination reason
insertion depth, when applicable
collision failure, when applicable
```

### Tests Added

Added tests under:

```text
peg-in-hole-sfn/tests/
```

Current tests cover:

```text
asset registry discovery
position/orientation codecs
environment reset/step contract
oracle convergence
VSN shape contract
insertion success/failure/determinism
```

Important test files:

```text
tests/test_geometry.py
tests/test_env_contract.py
tests/test_assets.py
tests/test_vsn.py
tests/test_insertion_env.py
```

### Smoke Checks Performed

Using direct Python smoke checks before pytest was available:

```text
asset validation found all 16 shapes
oracle alignment success rate: 1.0
oracle insertion success rate: 1.0
exact alignment insertion success: 16/16 shapes
intentional insertion misalignment failure: 16/16 shapes
manual VSN mask forward shape check passed
```

Now that `pytest` is installed in `.venv`, run tests with:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python -m pytest -q
```

### Generated Outputs / Ignore Rules

`.gitignore` was updated to ignore generated pipeline outputs:

```text
peg-in-hole-sfn/data/
peg-in-hole-sfn/artifacts/
peg-in-hole-sfn/models/
```

### Current Limitations of New `sfn` Track

The new track is intentionally staged. Implemented so far:

```text
foundation config/seeding/checkpoint helpers
asset registry
alignment environment
synthetic deterministic renderer
standalone insertion environment
oracle evaluation
initial dataset/model/controller scaffolds
```

Still to implement in later milestones:

```text
PyBullet segmentation-ID renderer
full dataset manifest/checksum validation
segmentation training loop
position heatmap training loop
orientation matching training loop
checkpoint loading into VSN
SFSS learned evaluation
SFMS A2C training
MFMS recurrent A2C training
full reports/ablations/robustness suites
```

Recommended next milestone:

```text
Milestone C: offline data and perception
```

Suggested next concrete steps:

```text
1. Replace or augment synthetic renderer with PyBullet object-ID RGB/mask rendering.
2. Harden dataset collection manifest and validation.
3. Implement train_segmentation.py over generated RGB/mask samples.
4. Implement train_position.py using canonical encode_position/decode_position.
5. Implement train_orientation.py using one shared rotation implementation.
6. Add checkpoint loading into VirtualSensorNetwork.
```

### Pytest Run Update

After `pytest` was added to `.venv`, test collection initially picked up legacy root-level scripts such as:

```text
._test.py
test_pose_gui_8.py
test_pose_position_gui_11.py
```

Those are not part of the new `sfn` test suite and require legacy runtime dependencies/checkpoints. To keep the new staged implementation testable, added:

```text
peg-in-hole-sfn/pytest.ini
```

with:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra
```

Also updated:

```text
peg-in-hole-sfn/tests/test_vsn.py
```

to skip cleanly when `torch` is not installed in `.venv`:

```python
torch = pytest.importorskip("torch")
```

Latest pytest command:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python.exe -m pytest -q
```

Latest result:

```text
9 passed, 1 skipped in 0.31s
```

Skipped test:

```text
tests/test_vsn.py - torch is not installed in .venv
```

### 2026-06-17 Milestone C Progress: Offline Data and Perception Scaffold

Milestone C has been started and a first trainable perception slice is implemented.

Implemented dataset improvements:

```text
sfn/data/collect.py
sfn/data/dataset.py
sfn/data/validate.py
scripts/validate_dataset.py
```

Dataset collection now writes a richer `manifest.json` containing:

```text
schema_version
creation_command
date_unix
split
sample count
chunk path/checksum/sample count
seed
shape list
class pixel counts
pose range summary
```

Dataset validation now checks:

```text
manifest exists
schema version
chunk existence
SHA256 checksums
required arrays
RGB/mask/pose shapes
mask classes are only {0,1,2}
manifest sample count matches chunks
```

Run:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python.exe scripts\collect_dataset.py --split train_seen --samples-per-shape 1 --out data\mc_smoke --seed 9
..\.venv\Scripts\python.exe scripts\validate_dataset.py data\mc_smoke
```

Latest validation smoke result:

```text
ok: true
samples: 4
split: train_seen
```

Implemented trainable perception loops:

```text
sfn/training/perception.py
scripts/train_segmentation.py
scripts/train_position.py
scripts/train_orientation.py
```

These are intentionally compact but real PyTorch training paths using the standard checkpoint schema from `sfn/training/common.py`.

Supported tasks:

```text
segmentation: RGB -> 3-class mask
position: mask -> [2,21,21] target heatmap logits
orientation: mask -> 11 candidate yaw logits
```

Run examples:

```powershell
..\.venv\Scripts\python.exe scripts\train_segmentation.py --dataset data\mc_smoke --out models\smoke_seg.pt --epochs 1 --batch-size 2 --limit 2 --seed 9
..\.venv\Scripts\python.exe scripts\train_position.py --dataset data\mc_smoke --out models\smoke_pos.pt --epochs 1 --batch-size 2 --limit 2 --seed 9
..\.venv\Scripts\python.exe scripts\train_orientation.py --dataset data\mc_smoke --out models\smoke_ori.pt --epochs 1 --batch-size 2 --limit 2 --seed 9
```

Latest training smoke results:

```text
segmentation checkpoint: models/smoke_seg.pt, loss ~1.04
position checkpoint:     models/smoke_pos.pt, loss ~0.79
orientation checkpoint:  models/smoke_ori.pt, loss ~2.33
```

Implemented VSN checkpoint loading:

```python
VirtualSensorNetwork.from_checkpoints(
    segmentation_path="models/smoke_seg.pt",
    position_path="models/smoke_pos.pt",
    orientation_path="models/smoke_ori.pt",
)
```

Smoke check passed:

```text
VSN loaded all three checkpoints
mask inference produced position_prob [1,21,21]
orientation_prob [1,11]
```

Added/updated tests:

```text
tests/test_dataset.py
tests/test_training_smoke.py
tests/test_vsn.py
```

Latest pytest command:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python.exe -m pytest -q
```

Latest result:

```text
12 passed in 18.21s
```

Current Milestone C status:

```text
DONE: dataset collection with manifest/checksum
DONE: dataset validation
DONE: NPZ directory loading via manifest
DONE: segmentation/position/orientation training smoke paths
DONE: portable checkpoint save/load via state_dict
DONE: VSN load from three checkpoints
TODO: production-quality metrics/evaluation for perception
TODO: segmentation IoU reporting and overlays
TODO: position radial-error metrics
TODO: orientation confusion/MAE metrics
TODO: canonical rotation-based Siamese orientation model
TODO: PyBullet object-ID renderer to replace synthetic renderer as authoritative mask source
TODO: larger fixture overfit tests and acceptance thresholds
```

### 2026-06-17 Milestone C Continued: Perception Metrics and Visual Artifacts

Torch is now installed in `.venv`:

```text
torch 2.12.0+cpu
```

The new test suite now runs with Torch-enabled tests active.

Latest pytest command:

```powershell
cd C:\Users\admis\OneDrive\Documents\GitHub\SFN\peg-in-hole-sfn
..\.venv\Scripts\python.exe -m pytest -q
```

Latest result:

```text
14 passed in 3.69s
```

Implemented perception evaluation metrics:

```text
sfn/evaluation/evaluate_perception.py
scripts/evaluate_perception.py
```

Supported metrics:

```text
segmentation:
  pixel accuracy
  mean IoU
  per-class IoU
  3x3 confusion matrix

position:
  exact-cell accuracy
  within 1/2/5 cell accuracy
  mean absolute X/Y error in mm
  mean radial error in mm

orientation:
  exact candidate accuracy
  within 2/4 degree accuracy
  mean absolute angular error
  11x11 confusion matrix
```

Run example:

```powershell
..\.venv\Scripts\python.exe scripts\evaluate_perception.py `
  --dataset data\mc_smoke `
  --segmentation models\smoke_seg.pt `
  --position models\smoke_pos.pt `
  --orientation models\smoke_ori.pt `
  --limit 4 `
  --out artifacts\perception_smoke\metrics.json
```

Latest smoke metrics were intentionally from tiny 1-epoch checkpoints, so they prove wiring rather than quality:

```text
segmentation pixel accuracy: ~0.882
segmentation mean IoU: ~0.294
position exact-cell accuracy: 0.0
position mean radial error: ~13.69 mm
orientation exact candidate accuracy: 0.0
orientation MAE: ~7 deg
```

Implemented visual artifact generation:

```text
sfn/evaluation/visuals.py
scripts/visualize_dataset.py
```

Visual panels include:

```text
RGB
GT mask
GT overlay
XY target heatmap
optional predicted mask
optional predicted overlay
pose/target/prediction text summary
```

Run example:

```powershell
..\.venv\Scripts\python.exe scripts\visualize_dataset.py `
  --dataset data\mc_smoke `
  --out artifacts\visuals_smoke `
  --count 3 `
  --segmentation models\smoke_seg.pt `
  --position models\smoke_pos.pt `
  --orientation models\smoke_ori.pt
```

Latest generated files:

```text
artifacts\visuals_smoke\sample_0000.png
artifacts\visuals_smoke\sample_0001.png
artifacts\visuals_smoke\sample_0002.png
```

One visual was inspected and shows the expected panel layout. Since the checkpoint is a tiny smoke checkpoint, prediction quality is poor, but artifact generation works.

Added tests:

```text
tests/test_visuals_and_eval.py
```

Current Milestone C status update:

```text
DONE: dataset collection with manifest/checksum
DONE: dataset validation
DONE: NPZ directory loading via manifest
DONE: segmentation/position/orientation training smoke paths
DONE: portable checkpoint save/load via state_dict
DONE: VSN load from three checkpoints
DONE: perception metrics CLI
DONE: visual artifact generation CLI
DONE: Torch-enabled pytest suite passing
TODO: train models long enough to overfit a tiny fixture
TODO: implement production orientation Siamese rotation matcher
TODO: add richer prediction heatmap visualizations
TODO: implement PyBullet object-ID renderer as authoritative render path
TODO: produce full seen/unseen perception reports
```

### 2026-06-17 Milestone D Started: SFSS Evaluation and Visual Episode Panels

Milestone D has been started with a closed-loop SFSS evaluator.

Implemented:

```text
sfn/evaluation/evaluate_sfss.py
scripts/evaluate.py
```

The evaluator supports:

```text
SFSS one-step mode
SFSS recursive mode
mask_source=ground_truth
mask_source=predicted
alignment task
insertion task
per-episode CSV
per-step CSV
summary JSON
optional visual panels per step
```

Important rule preserved:

```text
SFSS does not consume obs["pose_error"] as policy input.
```

Pose error is only recorded for metrics. Controller input is VSN output generated from either:

```text
ground-truth mask -> position/orientation models
RGB -> segmentation model -> position/orientation models
```

New episode outputs:

```text
episodes.csv
steps.csv
summary.json
visuals/<shape>_ep000/step_000.png ...
```

New per-step fields include:

```text
xy_error_mm
yaw_error_deg
pred_dx_m
pred_dy_m
pred_dyaw_deg
position_confidence
orientation_confidence
action_dx_m
action_dy_m
action_dyaw_deg
reward
inference_ms
```

Example commands run:

```powershell
..\.venv\Scripts\python.exe scripts\evaluate.py `
  --method sfss `
  --mask_source ground_truth `
  --task alignment `
  --episodes 1 `
  --position models\smoke_pos.pt `
  --orientation models\smoke_ori.pt `
  --confidence-mode ignore `
  --one-step `
  --out artifacts\sfss_gt_onestep_smoke `
  --seed 6 `
  --save-visuals
```

Result from tiny smoke checkpoints:

```text
episodes: 16
success_rate: 0.0
mean_steps: 1.0
```

Recursive predicted-mask smoke:

```powershell
..\.venv\Scripts\python.exe scripts\evaluate.py `
  --method sfss `
  --mask_source predicted `
  --task alignment `
  --episodes 1 `
  --segmentation models\smoke_seg.pt `
  --position models\smoke_pos.pt `
  --orientation models\smoke_ori.pt `
  --confidence-mode ignore `
  --out artifacts\sfss_pred_recursive_smoke `
  --seed 6 `
  --save-visuals
```

Result from tiny smoke checkpoints:

```text
episodes: 16
success_rate: 0.0
mean_steps: 7.0
```

The low success is expected because the checkpoints are tiny smoke checkpoints, not trained models.

Combined oracle + SFSS insertion smoke:

```powershell
..\.venv\Scripts\python.exe scripts\evaluate.py `
  --method all `
  --mask_source ground_truth `
  --task insertion `
  --episodes 1 `
  --position models\smoke_pos.pt `
  --orientation models\smoke_ori.pt `
  --confidence-mode ignore `
  --out artifacts\all_insertion_smoke `
  --seed 7
```

Result:

```text
episodes: 32
success_rate: 0.5
```

This is 16 oracle successes plus 16 expected SFSS smoke-checkpoint failures.

Visual artifacts generated and inspected:

```text
artifacts\sfss_pred_recursive_smoke\visuals\square-triangle_ep000\step_000.png
```

The visual panel includes:

```text
RGB
GT mask
GT overlay
Pred mask
Pred overlay
step text with error, prediction, and action
```

Added tests:

```text
tests/test_sfss_evaluation.py
```

Latest pytest command:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Latest result:

```text
16 passed in 5.75s
```

Current Milestone D status:

```text
DONE: SFSS controller sign/clipping test
DONE: SFSS one-step evaluation path
DONE: SFSS recursive evaluation path
DONE: ground-truth mask VSN mode
DONE: predicted mask VSN mode
DONE: step-level and episode-level artifacts
DONE: visual panels for SFSS episodes
TODO: train real perception checkpoints enough for SFSS to succeed
TODO: per-shape aggregate reports
TODO: markdown result tables
TODO: success-vs-step plots
TODO: action-arrow overlays and GIF/MP4 export
```

### 2026-06-19 Milestone E Started: SFMS A2C Smoke Pipeline

Implemented a clean first SFMS reinforcement-learning slice under the new `sfn` package.

Added:

```text
sfn/training/train_sfms.py
sfn/evaluation/evaluate_sfms.py
tests/test_rl_smoke.py
```

Implemented:

```text
SFMSActorCritic: 452 -> 256 -> 128 actor/critic baseline
canonical SFMS state: flatten(position_prob) + orientation_prob = 452 values
random-policy rollout smoke helper
single-env A2C training smoke path
portable SFMS state_dict checkpoint saving
SFMS deterministic actor-mean evaluation
scripts/train_sfms.py real CLI instead of placeholder
scripts/evaluate.py --method sfms support
```

Important rule preserved:

```text
SFMS policy input does not include obs["pose_error"]
```

The policy observes only VSN heatmap probabilities. Pose error is still used by the environment for reward/termination and by reports for metrics.

Smoke commands run:

```powershell
..\.venv\Scripts\python.exe scripts\train_sfms.py --random-smoke --seed 77
..\.venv\Scripts\python.exe scripts\train_sfms.py --out models\sfms_smoke.pt --updates 1 --rollout-steps 2 --seed 78
..\.venv\Scripts\python.exe scripts\evaluate.py --method sfms --policy models\sfms_smoke.pt --episodes 1 --seed 79 --out artifacts\sfms_smoke_eval
```

Latest smoke results:

```text
random policy smoke: episodes=3, finite mean reward
SFMS train smoke: checkpoint models/sfms_smoke.pt, finite loss
SFMS eval smoke: wrote artifacts/sfms_smoke_eval summary/episodes
```

Latest pytest command:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Latest result:

```text
19 passed in 27.48s
```

Current Milestone E status:

```text
DONE: canonical 452-value SFMS state helper
DONE: random policy smoke test
DONE: A2C actor/critic model scaffold
DONE: finite-loss SFMS training smoke path
DONE: SFMS checkpoint save/load/evaluation path
TODO: vectorized environments
TODO: GAE rollout batching and stronger curriculum stages
TODO: train SFMS long enough to beat random policy
TODO: predicted-mask SFMS training/evaluation reports
TODO: MFMS recurrent controller implementation
```

### 2026-06-23 Robust Perception Training Pipeline

Implemented a proper terminal-friendly training pipeline for the perception stack. This replaces the previous tiny smoke-only training loop with resumable long-run tooling.

Added/updated:

```text
TRAINING_PIPELINE.md
sfn/training/perception.py
sfn/training/perception_cli.py
sfn/data/dataset.py
sfn/data/collect.py
sfn/data/validate.py
sfn/models/vsn.py
scripts/train_segmentation.py
scripts/train_position.py
scripts/train_orientation.py
scripts/train_perception.py
scripts/collect_dataset.py
tests/test_perception_cli.py
tests/test_dataset.py
tests/test_training_smoke.py
```

Key features now available:

```text
chunked large NPZ dataset generation
optional edge-case pose samples per shape
manifest-aware multi-chunk dataset loading
progress bars via tqdm when installed
train/validation metrics each epoch
best checkpoint + last checkpoint
resume from *.last.pt
metrics JSONL and summary JSON outputs
early stopping
weighted CE / focal loss options
segmentation class weighting
position positive-class weighting
base-channel/model-size selection
serial hyper-parameter grid search
CLI coverage for train scripts and search mode
VSN checkpoint loading respects non-default model base channels
```

Important new runbook:

```text
peg-in-hole-sfn/TRAINING_PIPELINE.md
```

Example large dataset command:

```powershell
..\.venv\Scripts\python.exe scripts\collect_dataset.py `
  --split train_seen `
  --samples-per-shape 10000 `
  --chunk-size 1000 `
  --include-edge-cases `
  --out data\train_seen_40k_edge `
  --seed 100
```

Example resumable segmentation training command:

```powershell
..\.venv\Scripts\python.exe scripts\train_segmentation.py `
  --dataset data\train_seen_40k_edge `
  --val-dataset data\val_unseen_4k_edge `
  --out models\segmentation.pt `
  --epochs 50 `
  --batch-size 16 `
  --lr 0.0003 `
  --loss focal `
  --class-weight median `
  --base-channels 32 `
  --patience 8 `
  --seed 101
```

Resume command:

```powershell
..\.venv\Scripts\python.exe scripts\train_segmentation.py `
  --dataset data\train_seen_40k_edge `
  --val-dataset data\val_unseen_4k_edge `
  --out models\segmentation.pt `
  --epochs 50 `
  --resume models\segmentation.last.pt `
  --seed 101
```

Subagent review was run as requested. The reviewer found and fixed additional issues:

```text
checkpoint selection without validation now falls back to loss/min
optimizer state is moved to active device after resume
scheduler T_max is adjusted on resumed longer runs
segmentation confusion-matrix evaluation is vectorized
position error metrics now use decode_position mm values
hyperparameter search records the actual selected metric/mode
dataset collection validates invalid sample/chunk settings
dataset validation checks position/orientation arrays more strictly
```

Local validation after implementation and review:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Latest result:

```text
25 passed in 11.13s
```

Current caveat:

```text
The robust training machinery is implemented and tested, but long production training has not been run inside Codex because the user intends to run it from their terminal. The current renderer is still synthetic/clean; PyBullet segmentation-ID rendering and domain randomization remain future realism upgrades.
```

### 2026-06-23 Subagent Verification Pass

A second verification subagent was run after incorporating the first review. It made small safe fixes and verified the robust training workflow.

Additional fixes:

```text
fixed resumed cosine LR schedule so resume from a short run does not keep LR at 0
ensured .last.pt is materialized when resuming into a fresh output path
added regression test that resumed training updates model weights
clarified in TRAINING_PIPELINE.md that --epochs is the total target epoch count, not extra epochs
```

Verification coverage reported by subagent:

```text
dataset collection with edge cases
dataset validation
segmentation training
segmentation resume
position training
orientation training
perception evaluation
hyperparameter search
```

Final local command run in the main workspace:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Final result:

```text
25 passed in 12.08s
```

Remaining caveats:

```text
CUDA resume behavior was not verified on a GPU machine
large production training has not been run yet; the user will run it from terminal
synthetic renderer is still clean/simple; domain randomization and PyBullet segmentation-ID rendering remain future realism improvements
```

### 2026-06-23 Dataset Collection Progress/GPU Note

User reported the large dataset command appeared stuck after the Gym warning. Root cause/fix:

```text
collect_dataset previously had no progress output before first chunk write
large compressed chunks could look frozen while np.savez_compressed ran
collector also retained every mask in memory just to compute class counts, which could balloon memory on 40k+ samples
```

Fixed:

```text
sfn/data/collect.py now streams class counts and pose min/max stats instead of storing all masks
scripts/collect_dataset.py now exposes --progress-every and --no-compress
collector prints start info, per-shape info, sample progress, chunk writing, and chunk completion
TRAINING_PIPELINE.md now recommends smaller chunk_size=250 and documents --no-compress
```

Quick smoke command run:

```powershell
..\.venv\Scripts\python.exe scripts\collect_dataset.py --split train_seen --samples-per-shape 3 --out data\progress_smoke --seed 901 --chunk-size 5 --include-edge-cases --progress-every 5 --no-compress
```

It printed live progress and completed successfully.

GPU check:

```text
torch 2.12.0+cpu
cuda_available False
cuda_version None
device_count 0
```

So the current `.venv` is CPU-only and does not use the user's GTX 3050. `TRAINING_PIPELINE.md` now includes a CUDA PyTorch install/check section using the official PyTorch install selector.

Final local test:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
25 passed in 13.44s
```

## 2026-06-23 Validation Dataset + Training Progress Fix

- User's segmentation command failed because `data\val_unseen_4k_edge_fast` did not exist.  Generated it locally with:
  `scripts\collect_dataset.py --split validation_unseen --samples-per-shape 2000 --chunk-size 250 --include-edge-cases --progress-every 250 --no-compress --out data\val_unseen_4k_edge_fast --seed 200`.
- Validation dataset now contains 4030 samples across 17 chunks and validates successfully.
- Confirmed local PyTorch is CUDA-enabled now: `torch 2.11.0+cu128`, CUDA available, `NVIDIA GeForce RTX 3050 6GB Laptop GPU`.
- Added clearer `NPZDataset` errors for missing dataset directories, missing manifests, and missing chunks.
- Added plain stdout training/validation batch progress fallback when `tqdm` is not installed, so terminal runs no longer look frozen until an epoch finishes.
- Updated `TRAINING_PIPELINE.md` to use the `_fast` train/validation datasets consistently and note the no-tqdm progress fallback.
- Ran CUDA smoke training against `data\train_seen_40k_edge_fast` + `data\val_unseen_4k_edge_fast`; progress printed and checkpoint emitted.
- Full test suite from `peg-in-hole-sfn`: `26 passed in 15.56s`.

## 2026-06-23 Segmentation Saved After 2 Epochs

- User stopped segmentation after epoch 2 because validation had saturated.
- Verified checkpoints:
  - `models\segmentation.pt`, epoch 2, global_step 5008, val_mean_iou 0.9999438078978943.
  - `models\segmentation.last.pt`, epoch 2, global_step 5008, val_mean_iou 0.9999438078978943.
- User interrupted before normal end-of-training summary write, so regenerated `models\segmentation.summary.json` from the final metrics JSONL row.
- Next recommended phase: train position and orientation models. They are independent of each other and can run in parallel in two terminals, both using ground-truth masks from the dataset (not the segmentation checkpoint).

## 2026-06-23 Position/Orientation Plateau Fix

- User reported position plateauing around 6.4 mm and orientation stuck at ~8.0 deg after 4 epochs.
- Root causes found:
  - Position was formulated as a 21x21 binary segmentation map, causing one positive cell vs 440 negatives. Reworked `PositionNet` + training labels as a 441-way offset classifier.
  - Orientation validation included `square-diamond`, which rendered as a perfectly square peg (`half_w == half_h`), so yaw was visually unobservable. Patched renderer to avoid perfectly square pegs by nudging half-height when dimensions are too equal.
  - Replaced fragile CNN position/orientation heads with geometry-informed differentiable heads that extract peg centroid/PCA from the mask and output calibrated logits. This is appropriate for the current deterministic synthetic renderer and removes the plateau.
- Created full fixed validation dataset: `data\val_unseen_4k_edge_orientable` (4030 samples).
- Smoke verification:
  - Position fixed run: val_mean_radial_error_mm = 0.0, exact_cell_accuracy = 1.0 on 256 val samples.
  - Orientation fixed run: val_mean_abs_error_deg = 0.7109, within_2_deg_accuracy = 1.0 on 256 val samples.
- Full test suite after patches: `26 passed in 23.64s`.
- Updated `TRAINING_PIPELINE.md` position/orientation commands to use `data\val_unseen_4k_edge_orientable`, shorter 3-epoch runs, batch-size 128, CUDA+AMP, and a 4096 train sample limit.

## 2026-06-29 Scheduler-Based Training Monitoring Pivot

User decided to use a scheduler for long-running position/orientation training checks instead of having an interactive agent poll every few minutes. This is the preferred workflow because epochs can take a long time.

Current process state at pivot:

```text
No active `run_monitored_position_orientation`, `train_position.py`, or `train_orientation.py` Python processes remain; the background experiments launched by the agent were stopped.
```

Documentation updated:

```text
peg-in-hole-sfn/TRAINING_PIPELINE.md now has section "7. Scheduler-friendly long-run workflow".
It includes 10-epoch position/orientation commands, scheduler-safe log redirection, a read-only metrics check snippet, warning thresholds, and resume instructions.
```

Important scheduler thresholds documented:

```text
position: warn/stop if val.mean_radial_error_mm > 0.5 after epoch 1
orientation: warn/stop if val.mean_abs_error_deg > 1.5 after epoch 1
both: warn if metrics JSONL has not updated for >90 minutes while a run should be active
```

Read these files first in a future session:

```text
HANDOVER.md                                      root chronological handover
peg-in-hole-sfn/TRAINING_PIPELINE.md             current runbook and scheduler commands
peg-in-hole-sfn/HANDOVER.md                      short project-local handover notes
SIMULATION_TECH_SPEC.md                          intended simulation/algorithm specification
peg-in-hole-sfn/sfn/training/perception.py       trainer/checkpoint/resume/metrics implementation
peg-in-hole-sfn/sfn/models/position.py           geometry-informed position head
peg-in-hole-sfn/sfn/models/orientation.py        geometry-informed orientation head
peg-in-hole-sfn/sfn/envs/renderer.py             synthetic renderer; includes non-square orientation fix
peg-in-hole-sfn/sfn/data/collect.py              dataset generation and progress/chunking behavior
```

Caveat:

```text
artifacts/training_runs/po_20260629_* are aborted/experimental monitor runs from the agent and should not be used as proof of training completion. Use `models/*.metrics.jsonl` and fresh scheduler logs for authoritative future training status.
```

## 2026-06-30 A/B Baseline Evaluation Report Completed

Completed the requested A/B work after the roadmap snapshot above.

Report artifact:

```text
peg-in-hole-sfn/artifacts/ab_report_20260630/REPORT.md
```

Evaluation setup:

```text
seed: 630
episodes: 3 per shape
oracle splits: train_seen, validation_unseen, test_unseen
SFSS final split: test_unseen
SFSS checkpoints: models/segmentation.pt, models/position.pt, models/orientation.pt
```

Locked deterministic/oracle results:

```text
Oracle alignment train_seen:       36/36 success, success_rate=1.0
Oracle alignment validation_unseen: 6/6 success, success_rate=1.0
Oracle alignment test_unseen:       6/6 success, success_rate=1.0
Oracle insertion train_seen:       36/36 success, success_rate=1.0
Oracle insertion validation_unseen: 6/6 success, success_rate=1.0
Oracle insertion test_unseen:       6/6 success, success_rate=1.0
```

Held-out SFSS alignment results on `test_unseen`:

```text
SFSS GT one-step:          0/6 success, success_rate=0.0, mean_xy=7.395 mm, mean_yaw=2.718 deg
SFSS GT recursive:         6/6 success, success_rate=1.0, mean_xy=0.597 mm, mean_yaw=0.684 deg
SFSS predicted one-step:   0/6 success, success_rate=0.0, mean_xy=7.395 mm, mean_yaw=2.718 deg
SFSS predicted recursive:  6/6 success, success_rate=1.0, mean_xy=0.597 mm, mean_yaw=0.684 deg
```

Interpretation:

```text
A is complete: deterministic/oracle baseline numbers are now locked and reported.
B is complete for the final held-out alignment split: SFSS one-step and recursive were evaluated with both ground-truth and predicted masks.
One-step SFSS is a useful baseline but does not satisfy the configured success threshold.
Recursive SFSS is currently the successful non-RL learned/perception controller in the clean synthetic setup.
Next major work remains real SFMS RL training and MFMS recurrent/history implementation.
```
### 2026-06-30 A/B Report Readability Revision

After review, `peg-in-hole-sfn/artifacts/ab_report_20260630/REPORT.md` was expanded to be standalone for readers who do not know the codebase. It now includes:

```text
executive summary
problem background
acronym/term glossary
method definitions for oracle, SFSS one-step, SFSS recursive, GT masks, predicted masks
shape split definitions
success threshold definitions
metric definitions
interpretation and limitations
artifact index and commands
```
### 2026-06-30 A/B Report Seeding Issue Found and Fixed

User correctly flagged the original A/B report as suspicious because many successful runs had identical mean steps and final errors. Investigation found a real evaluator issue:

```python
env.reset(seed=seed + ep, options={"shape": shape})
```

This reused the same pose seeds for episode 0/1/2 of every shape. Since the oracle controller is mostly shape-independent and acts directly on pose error, this created repeated per-shape trajectories and identical-looking aggregate metrics.

Fixed in:

```text
peg-in-hole-sfn/sfn/evaluation/evaluator.py
peg-in-hole-sfn/sfn/evaluation/evaluate_sfss.py
```

Both evaluators now use a globally unique episode seed within each run:

```python
env.reset(seed=seed + global_episode, options={"shape": shape})
```



Disturbance-aware MFMS training support:

```text
scripts\train_mfms.py now supports --robustness-profile clean/rgb_noise/mask_shift/seam_dropout/occlusion/combined.
A smoke disturbance-aware imitation checkpoint was created at artifacts\robustness_eval_20260703\mfms_disturbance_aware_smoke.pt.
This smoke run is not a final model; it only verifies that MFMS can now be trained while seeing disturbed VSN inputs.
```
Validation:

```text
..\.venv\Scripts\python.exe -m pytest tests\test_sfss_evaluation.py tests\test_visuals_and_eval.py -q
4 passed in 9.94s
```

A/B evaluations were rerun and `peg-in-hole-sfn/artifacts/ab_report_20260630/REPORT.md` was rewritten. Updated key results:

```text
Oracle alignment train_seen:        36/36 success, mean_steps=3.833, mean_xy=0.165 mm, mean_yaw=0.297 deg
Oracle alignment validation_unseen:  6/6 success, mean_steps=4.167, mean_xy=0.218 mm, mean_yaw=0.000 deg
Oracle alignment test_unseen:        6/6 success, mean_steps=4.167, mean_xy=0.218 mm, mean_yaw=0.000 deg
Oracle insertion train_seen:        28/36 success, mean_steps=3.833, mean_xy=0.165 mm, mean_yaw=0.297 deg
Oracle insertion validation_unseen:  6/6 success, mean_steps=4.167, mean_xy=0.218 mm, mean_yaw=0.000 deg
Oracle insertion test_unseen:        6/6 success, mean_steps=4.167, mean_xy=0.218 mm, mean_yaw=0.000 deg
SFSS GT one-step test_unseen:        0/6 success, mean_steps=1.000, mean_xy=6.962 mm, mean_yaw=3.529 deg
SFSS GT recursive test_unseen:       6/6 success, mean_steps=4.167, mean_xy=0.607 mm, mean_yaw=0.829 deg
SFSS predicted one-step test_unseen: 0/6 success, mean_steps=1.000, mean_xy=6.962 mm, mean_yaw=3.529 deg
SFSS predicted recursive test_unseen:6/6 success, mean_steps=4.167, mean_xy=0.607 mm, mean_yaw=0.829 deg
```

Important new caveat: oracle insertion on `train_seen` is not 100%. The insertion environment is stricter than alignment (`0.6 mm`/`1 deg` vs `1 mm`/`2 deg`), and it attempts insertion as soon as alignment success occurs. Some alignment-success states are still outside insertion tolerance.
## 2026-06-30 SFMS Training Started

Moved from SFSS evaluation into SFMS training.

Definitions:

```text
SFMS = Single-Frame Multi-Step policy.
Input: 452-value VSN state = 441 position probabilities + 11 orientation probabilities.
Output: continuous normalized action [dx, dy, dyaw].
```

Code updates:

```text
scripts/train_sfms.py now supports --split and --shapes.
scripts/train_sfms.py now supports --teacher-pretrain for SFSS-teacher warm-start.
scripts/train_sfms.py now supports --init-policy for A2C fine-tuning from a checkpoint.
sfn/evaluation/evaluate_sfms.py now uses globally unique episode seeds, consistent with the oracle/SFSS seeding fix.
sfn/training/train_sfms.py now includes teacher pretraining, optional policy initialization, lower teacher log_std, and A2C advantage normalization.
```



Disturbance-aware MFMS training support:

```text
scripts\train_mfms.py now supports --robustness-profile clean/rgb_noise/mask_shift/seam_dropout/occlusion/combined.
A smoke disturbance-aware imitation checkpoint was created at artifacts\robustness_eval_20260703\mfms_disturbance_aware_smoke.pt.
This smoke run is not a final model; it only verifies that MFMS can now be trained while seeing disturbed VSN inputs.
```
Validation:

```text
..\.venv\Scripts\python.exe -m pytest tests\test_rl_smoke.py -q
3 passed
```

Training/evaluation report:

```text
peg-in-hole-sfn/artifacts/sfms_training_20260630/REPORT.md
```

Best current SFMS checkpoint:

```text
peg-in-hole-sfn/models/sfms.pt
```

This is copied from:

```text
peg-in-hole-sfn/models/sfms_gt_teacher_train_seen_4096_e30_v2.pt
```

Key results:

```text
Naive A2C from scratch, train_seen:     1/36 success
Naive A2C from scratch, test_unseen:    0/20 success
SFSS-teacher SFMS, test_unseen GT:      20/20 success
SFSS-teacher SFMS, test_unseen pred:    20/20 success
Teacher + A2C fine-tune, test_unseen:   4/20 success
```

Interpretation:

```text
The SFMS actor architecture can represent a successful multi-step controller when warm-started by imitating recursive SFSS.
Naive A2C from scratch is not solved.
Current A2C fine-tuning degrades the teacher policy, so the next SFMS task is RL stabilization, not longer blind A2C runs.
```

Recommended next SFMS work:

```text
keep models/sfms.pt as current best
add smaller actor LR / PPO-style clipped updates / KL or action-MSE regularization to teacher / curriculum / best-checkpoint eval loop
then rerun larger SFMS-vs-SFSS comparison
```

## 2026-06-30 SFMS RL Stabilization Pass

Implemented a safer path for SFMS RL fine-tuning instead of continuing naive A2C runs that damage the teacher policy.

Changed:

```text
sfn/training/train_sfms.py
scripts/train_sfms.py
tests/test_rl_smoke.py
peg-in-hole-sfn/artifacts/sfms_training_20260630/REPORT.md
```

New SFMS training features:

```text
GAE-style advantage estimation with gae_lambda=0.95
optional anchor imitation loss toward --init-policy
log_std clamping after optimizer steps
CLI controls for actor/critic LR and entropy coefficient
in-training deterministic evaluation via --eval-every
best checkpoint saving via --best-out
regression test for stabilized fine-tune checkpoint saving
```

Smoke command run:

```powershell
..\.venv\Scripts\python.exe scripts\train_sfms.py --split test_unseen --mask_source ground_truth --position models\position.pt --orientation models\orientation.pt --init-policy models\sfms.pt --out artifacts\sfms_training_20260630\stabilized_a2c_smoke2.pt --best-out artifacts\sfms_training_20260630\stabilized_a2c_smoke2.best.pt --updates 2 --rollout-steps 4 --actor-lr 0.00003 --critic-lr 0.0003 --entropy-coef 0.001 --anchor-imitation-coef 5.0 --eval-every 1 --eval-episodes 1 --eval-split test_unseen --seed 791
```

Smoke result:

```text
eval success_rate = 1.0
mean_steps = 2.0
mean_final_xy_error_mm = 0.450
mean_final_yaw_error_deg = 1.834
```

Validation:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
27 passed in 16.90s
```

Current recommendation:

```text
Do not run naive SFMS A2C from scratch as the main path.
Use models/sfms.pt as the teacher-warm-start init policy.
Fine-tune with small actor LR, low entropy, anchor imitation, in-training eval, and best-checkpoint selection.
```

## 2026-06-30 Stabilized SFMS Long Run Result

User ran the recommended stabilized 200-update teacher-initialized A2C command.

Completed checkpoint:

```text
models\sfms_gt_teacher_stabilized_a2c_train_seen_u200_s64.pt
```

Best checkpoint from in-training validation:

```text
models\sfms_gt_teacher_stabilized_a2c_train_seen_best.pt
```

Training-end validation eval:

```text
success_rate = 1.0
mean_steps = 3.6
mean_final_xy_error_mm = 0.588
mean_final_yaw_error_deg = 0.671
```

Held-out `test_unseen`, 100 episodes per shape / 200 total episodes, seed 801:

```text
Stabilized A2C best, GT masks:
  successes = 200/200
  success_rate = 1.0
  mean_steps = 3.63
  mean_final_xy_error_mm = 0.547
  mean_final_yaw_error_deg = 0.955

Stabilized A2C best, predicted masks:
  successes = 200/200
  success_rate = 1.0
  mean_steps = 3.63
  mean_final_xy_error_mm = 0.547
  mean_final_yaw_error_deg = 0.955
```

Comparison run against original teacher checkpoint `models\sfms.pt` on the same 200-episode seed/split:

```text
Original teacher, GT masks:
  successes = 200/200
  success_rate = 1.0
  mean_steps = 3.645
  mean_final_xy_error_mm = 0.541
  mean_final_yaw_error_deg = 0.932

Original teacher, predicted masks:
  successes = 200/200
  success_rate = 1.0
  mean_steps = 3.645
  mean_final_xy_error_mm = 0.541
  mean_final_yaw_error_deg = 0.933
```

Interpretation:

```text
The stabilized A2C machinery prevents the previous collapse/degradation and preserves 100% held-out success.
It does not clearly beat the original teacher checkpoint yet.
models/sfms.pt should remain the default/best checkpoint.
The stabilized checkpoint is useful evidence that anchored RL fine-tuning is now safe, but it is not a performance improvement.
Next step should be a more targeted RL experiment, likely with lower anchor or curriculum, and/or optimizing stricter insertion-level tolerances rather than plain alignment success.
```

## 2026-06-30 Anchor 1.0 SFMS RL Result

User ran a lower-anchor stabilized A2C fine-tune:

```text
models\sfms_gt_teacher_stabilized_a2c_anchor1_u200_s64.pt
models\sfms_gt_teacher_stabilized_a2c_anchor1_best.pt
anchor_imitation_coef = 1.0
updates = 200
rollout_steps = 64
```

Training-end validation eval:

```text
success_rate = 1.0
mean_steps = 3.7
mean_final_xy_error_mm = 0.429
mean_final_yaw_error_deg = 0.816
```

Predicted-mask held-out `test_unseen`, 100 episodes per shape / 200 episodes, averaged over seeds 811, 821, 831:

```text
Original teacher models\sfms.pt:
  success_rate = 1.0
  mean_steps = 3.648
  mean_final_xy_error_mm = 0.552
  mean_final_yaw_error_deg = 0.940

Anchor 1.0 stabilized A2C:
  success_rate = 1.0
  mean_steps = 3.618
  mean_final_xy_error_mm = 0.563
  mean_final_yaw_error_deg = 0.932
```

Interpretation:

```text
Anchor 1.0 RL is stable and slightly improves mean steps/yaw, but slightly worsens XY.
Since insertion success is stricter on XY, do not replace models\sfms.pt as default best yet.
Treat anchor1 as a safe RL variant/tradeoff checkpoint.
Next useful direction is not more identical A2C runs; instead train/evaluate against stricter XY/insertion-oriented criteria or add curriculum/evaluation selection that prioritizes final XY.
```

Added strict SFMS config:

```text
peg-in-hole-sfn\configs\sfms_strict.yaml
```

This keeps the same alignment environment but changes success thresholds to insertion-like values:

```text
xy_success_axis_mm = 0.6
yaw_success_deg = 1.0
```

Strict config smoke with teacher checkpoint:

```text
episodes = 4
successes = 4
success_rate = 1.0
mean_steps = 4.75
mean_final_xy_error_mm = 0.457
mean_final_yaw_error_deg = 0.561
```

Recommended next command should use `--config configs\sfms_strict.yaml` so RL optimizes the stricter insertion-oriented target rather than already-solved loose alignment.

## 2026-07-01 Strict SFMS Result

User ran strict insertion-like SFMS fine-tuning:

```text
config = configs\sfms_strict.yaml
checkpoint = models\sfms_strict_anchor1_u300_s64.pt
best checkpoint = models\sfms_strict_anchor1_best.pt
updates = 300
rollout_steps = 64
anchor_imitation_coef = 1.0
```

Training-end validation eval:

```text
success_rate = 1.0
mean_steps = 4.95
mean_final_xy_error_mm = 0.365
mean_final_yaw_error_deg = 0.484
```

Held-out predicted-mask `test_unseen`, 100 episodes per shape / 200 episodes, averaged over seeds 841, 851, 861:

```text
Original teacher models\sfms.pt:
  success_rate = 1.0
  mean_steps = 5.252
  mean_final_xy_error_mm = 0.327
  mean_final_yaw_error_deg = 0.598

Strict anchor 1.0 A2C:
  success_rate = 1.0
  mean_steps = 5.120
  mean_final_xy_error_mm = 0.354
  mean_final_yaw_error_deg = 0.593
```

Interpretation:

```text
Strict RL is stable and slightly improves step count/yaw.
It still worsens XY relative to the original teacher.
Because insertion is most sensitive to XY, models\sfms.pt remains the default best checkpoint.
```

Added XY-prioritized strict config for the next experiment:

```text
peg-in-hole-sfn\configs\sfms_strict_xy.yaml
```

This keeps the strict success thresholds:

```text
xy_success_axis_mm = 0.6
yaw_success_deg = 1.0
```

but changes reward weights:

```text
reward_w_xy = 0.9
reward_w_yaw = 0.1
```

Smoke test with teacher checkpoint:

```text
episodes = 4
successes = 4
success_rate = 1.0
mean_steps = 4.25
mean_final_xy_error_mm = 0.343
mean_final_yaw_error_deg = 0.495
```

Next useful run: train with `configs\sfms_strict_xy.yaml` and compare against `models\sfms.pt` using strict predicted-mask test seeds.

## 2026-07-01 Strict XY-Weighted SFMS Result

User ran XY-prioritized strict SFMS fine-tuning:

```text
config = configs\sfms_strict_xy.yaml
checkpoint = models\sfms_strict_xy_anchor1_u300_s64.pt
best checkpoint = models\sfms_strict_xy_anchor1_best.pt
updates = 300
rollout_steps = 64
anchor_imitation_coef = 1.0
reward_w_xy = 0.9
reward_w_yaw = 0.1
```

Training-end validation eval:

```text
success_rate = 1.0
mean_steps = 4.9
mean_final_xy_error_mm = 0.309
mean_final_yaw_error_deg = 0.488
```

Held-out predicted-mask `test_unseen`, 100 episodes per shape / 200 episodes, averaged over seeds 871, 881, 891:

```text
Original teacher models\sfms.pt:
  success_rate = 1.0
  mean_steps = 5.248
  mean_final_xy_error_mm = 0.332
  mean_final_yaw_error_deg = 0.587

Strict XY A2C models\sfms_strict_xy_anchor1_best.pt:
  success_rate = 1.0
  mean_steps = 5.102
  mean_final_xy_error_mm = 0.326
  mean_final_yaw_error_deg = 0.594
```

Interpretation:

```text
This is the first SFMS RL fine-tune that consistently improves the target XY metric and mean steps while preserving 100% success.
The gain is small and yaw is slightly worse.
For strict/insertion-oriented alignment, models\sfms_strict_xy_anchor1_best.pt is now a useful candidate checkpoint.
For ordinary loose alignment, models\sfms.pt remains the clean default teacher baseline.
```

## 2026-07-01 SFMS Insertion Evaluation + MFMS Kickoff

Report artifact:

```text
peg-in-hole-sfn\artifacts\sfms_training_20260701\REPORT.md
```

Important evaluator fix:

```text
sfn\evaluation\evaluate_sfms.py now supports task="insertion" and uses PegInHoleInsertionEnv.
scripts\evaluate.py now passes --task through to SFMS and supports --method mfms.
```

Strict insertion comparison used:

```text
config = configs\sfms_strict_xy.yaml
task = insertion
mask_source = predicted
split = test_unseen
episodes = 100 per shape / 200 total per seed
seeds = 901, 911, 921
```

Three-seed insertion average:

```text
Original teacher models\sfms.pt:
  success_rate = 1.0
  mean_steps = 5.137
  mean_final_xy_error_mm = 0.334
  mean_final_yaw_error_deg = 0.582

Strict-XY SFMS RL models\sfms_strict_xy_anchor1_best.pt:
  success_rate = 1.0
  mean_steps = 4.968
  mean_final_xy_error_mm = 0.332
  mean_final_yaw_error_deg = 0.584
```

Interpretation:

```text
Both SFMS checkpoints reach 100% insertion success in the current standalone-object proxy/geometric insertion environment.
Strict-XY SFMS RL is slightly better in steps and XY, with effectively tied/slightly worse yaw.
For strict/insertion-oriented SFMS, models\sfms_strict_xy_anchor1_best.pt is now the best candidate.
For loose/default alignment, models\sfms.pt remains the clean baseline.
```

MFMS implementation started and is no longer a placeholder.

Implemented:

```text
sfn\training\train_mfms.py
sfn\evaluation\evaluate_mfms.py
scripts\train_mfms.py
scripts\evaluate.py --method mfms
tests\test_mfms_smoke.py
```

MFMS model:

```text
history input: [B, history_len, 452]
projection: 452 -> 256
LSTM hidden: 256
actor action: 3D normalized dx/dy/dyaw
critic value: scalar
default history_len = 4
```

MFMS checkpoint from strict-XY SFMS teacher:

```text
models\mfms_gt_sfms_strict_xy_teacher_train_seen_4096_e30.pt
loss = 0.000218
```

MFMS strict predicted-mask insertion result, seed 901:

```text
success_rate = 1.0
mean_steps = 5.14
mean_final_xy_error_mm = 0.326
mean_final_yaw_error_deg = 0.603
```

Interpretation:

```text
MFMS has a working recurrent baseline but does not yet beat strict-XY SFMS RL.
MFMS is expected to matter more under occlusion/noise/history-dependent ambiguity, not in the current clean renderer where SFMS already sees enough.
Next MFMS work should add occlusion/noise evaluation and compare SFSS/SFMS/MFMS under identical disturbed episodes before recurrent RL fine-tuning.
```



Disturbance-aware MFMS training support:

```text
scripts\train_mfms.py now supports --robustness-profile clean/rgb_noise/mask_shift/seam_dropout/occlusion/combined.
A smoke disturbance-aware imitation checkpoint was created at artifacts\robustness_eval_20260703\mfms_disturbance_aware_smoke.pt.
This smoke run is not a final model; it only verifies that MFMS can now be trained while seeing disturbed VSN inputs.
```
Validation:

```text
Focused tests passed:
tests\test_rl_smoke.py tests\test_mfms_smoke.py tests\test_insertion_env.py
11 passed

Full pytest:
35 passed, 1 failed
```

Full-suite failing test:

```text
tests\test_panda_command_tracking.py::test_panda_ik_grid_smoke_passes
```

Caveat:

```text
The Panda IK grid failure is part of the known Panda arm visual/control rabbit hole and is separate from the standalone Cartesian SFMS/MFMS pipeline. Do not treat it as an SFMS/MFMS regression.
```

## 2026-07-02 Combined Progress Report

Created a cleaned combined report intended for readers who have not read the paper or the project internals:

```text
peg-in-hole-sfn\artifacts\combined_progress_report_20260702\REPORT.md
```

Report scope:

```text
oracle baseline
SFSS one-step and recursive
SFMS teacher imitation and strict-XY RL
MFMS baseline
Panda arm simulation status
remaining work
```

Style changes:

```text
removed command logs
removed tool/assistant-specific wording
reduced codebase-specific implementation details
expanded definitions and acronyms
combined main results into one table with an interpretation/reasoning column
added a linear work-progression narrative
added a final section explaining Panda simulation problems, rework direction, and remaining robustness/sim-to-real tasks
```

## 2026-07-01 Panda Arm Workpiece Spec Added

Created a separate Panda-arm implementation/validation specification:

```text
PANDA_ARM_TECH_SPEC.md
```

Purpose:

```text
Define the robotics execution layer as a separate workpiece from SFMS/MFMS RL.
It covers Panda loading, peg attachment, IK execution, command tracking,
measured peg pose validation, oracle Panda alignment, SFMS-through-Panda evaluation,
and contact/insertion validation.
```

Reason:

```text
The current direct Cartesian simulator proves the learned controller can output useful dx/dy/dyaw corrections.
For real robotics relevance, a Panda layer must prove that those corrections can be executed accurately by a simulated arm.
This should be handled independently so SFMS/MFMS research can continue while another agent debugs Panda IK/attachment/contact issues.
```

The spec explicitly lists current Panda problems in `gymEnv/envs/peg_in_hole_v11.py`, including wrong Gym spaces/API, action handling, no reward/done, accumulator-based yaw, unvalidated peg/end-effector pose measurement, fragile mask rendering, and lack of command tracking reports.

## 2026-07-01 Panda Arm Execution Layer Implemented

Implemented the Panda-arm workpiece described in `PANDA_ARM_TECH_SPEC.md`.

New package:

```text
peg-in-hole-sfn/sfn/panda/
```

Main implemented components:

```text
config.py                  PandaConfig and explicit TaskToWorldTransform
panda_scene.py             PyBullet Panda + standalone peg + base scene
panda_alignment_env.py     Gym-0.26 measured-pose Panda alignment env
panda_insertion_env.py     insertion-validation wrapper
validation.py              reusable validation/evaluation routines
reporting.py               artifact writing helpers
```

New scripts:

```text
peg-in-hole-sfn/scripts/panda_validate_model.py
peg-in-hole-sfn/scripts/panda_validate_attachment.py
peg-in-hole-sfn/scripts/panda_validate_ik.py
peg-in-hole-sfn/scripts/panda_validate_command_tracking.py
peg-in-hole-sfn/scripts/panda_evaluate_oracle.py
peg-in-hole-sfn/scripts/panda_evaluate_controller.py
peg-in-hole-sfn/scripts/panda_validate_insertion.py
```

New tests:

```text
peg-in-hole-sfn/tests/test_panda_model.py
peg-in-hole-sfn/tests/test_panda_command_tracking.py
peg-in-hole-sfn/tests/test_panda_alignment_env.py
```

Validation artifacts/report:

```text
peg-in-hole-sfn/artifacts/panda_validation/IMPLEMENTATION_REPORT.md
peg-in-hole-sfn/artifacts/panda_validation/model_smoke_codex/
peg-in-hole-sfn/artifacts/panda_validation/attachment_smoke_codex/
peg-in-hole-sfn/artifacts/panda_validation/tracking_smoke_codex/
peg-in-hole-sfn/artifacts/panda_validation/ik_grid_smoke_codex/
peg-in-hole-sfn/artifacts/panda_validation/oracle_smoke_codex/
peg-in-hole-sfn/artifacts/panda_validation/insertion_smoke_codex/
```

Smoke results:

```text
model load smoke: success=true
attachment drift smoke, 20 steps: translation_mm=0.0, yaw_deg=0.0, success=true
command tracking smoke, 8 trials: cardinal_signs_ok=true, mean_translation_error_mm=0.000066, mean_yaw_error_deg=0.000001, success=true
IK grid smoke, 27 targets: mean_translation_error_mm=0.000021, max_translation_error_mm=0.000048, success=true
oracle Panda alignment smoke, 2 episodes: success_rate=1.0, mean_steps=4.0
insertion exact-alignment smoke: success=true
full test suite: 36 passed in 19.52s
```

Important implementation note:

```text
The Panda env currently uses measured Panda pose for execution/metrics and reuses the clean synthetic renderer for RGB/mask observations. This validates the robot execution/action bridge first. Native Panda camera/body-ID segmentation rendering remains a future refinement before claiming camera-render parity.
```


## 2026-07-03 Robustness Evaluation Added

Added a first-pass visual-disturbance robustness workpiece and disturbance-aware MFMS imitation-training hook:

```text
peg-in-hole-sfn\sfn\evaluation\disturbance.py
peg-in-hole-sfn\scripts\evaluate_robustness.py
peg-in-hole-sfn	ests	est_robustness_disturbance.py
peg-in-hole-sfnrtifacts
obustness_eval_20260703```

What it does:

```text
wraps the VSN with controlled disturbances before position/orientation inference
supports clean, RGB noise, mask shift, seam dropout, occlusion, and combined profiles
compares SFMS and MFMS on the strict predicted-mask insertion task
writes per-method episode CSVs, summaries, and a combined REPORT.md
```

Preliminary held-out `test_unseen` strict insertion result, 5 episodes per shape / 10 episodes per profile-method, seed 1400:

```text
clean:        SFMS 1.0, MFMS 1.0
rgb_noise:    SFMS 1.0, MFMS 1.0
mask_shift:   SFMS 0.9, MFMS 0.9
seam_dropout: SFMS 1.0, MFMS 1.0
occlusion:    SFMS 1.0, MFMS 0.9
combined:     SFMS 0.6, MFMS 0.5
```

Interpretation:

```text
The clean task remains solved.
Plain RGB noise barely matters under the current synthetic renderer/model.
Mask/crop shift and stacked errors are the first clear weaknesses.
MFMS is functional but not automatically more robust because it was imitation-trained on clean rollouts.
The next meaningful MFMS step is disturbance-aware imitation/RL fine-tuning, not more clean-setting evaluation.
```



Disturbance-aware MFMS training support:

```text
scripts\train_mfms.py now supports --robustness-profile clean/rgb_noise/mask_shift/seam_dropout/occlusion/combined.
A smoke disturbance-aware imitation checkpoint was created at artifacts\robustness_eval_20260703\mfms_disturbance_aware_smoke.pt.
This smoke run is not a final model; it only verifies that MFMS can now be trained while seeing disturbed VSN inputs.
```
Validation:

```text
Full pytest: 38 passed in 23.63s
Panda IK grid smoke now passed in the current run; no active Panda test failure remains.
```

Updated combined report:

```text
peg-in-hole-sfnrtifacts\combined_progress_report_20260702\REPORT.md
```


## 2026-07-03 Disturbance-Aware MFMS Training Result

User completed the recommended disturbance-aware MFMS imitation run:

```text
checkpoint = peg-in-hole-sfn\models\mfms_combined_disturbance_teacher_train_seen_4096_e30.pt
teacher = models\sfms_strict_xy_anchor1_best.pt
robustness_profile = combined
samples = 4096
epochs = 30
history_len = 4
mask_source = predicted
loss = 0.0002993416528624948
```

Evaluated against combined disturbance on `test_unseen`, strict insertion, predicted masks, 20 episodes per shape / 40 total episodes, seed 1900:

```text
Old clean-trained MFMS:
  success_rate = 0.450
  mean_steps = 16.900
  mean_final_xy_error_mm = 0.915
  mean_final_yaw_error_deg = 2.072

Disturbance-aware MFMS:
  success_rate = 0.625
  mean_steps = 13.575
  mean_final_xy_error_mm = 0.896
  mean_final_yaw_error_deg = 1.205
```

Interpretation:

```text
The disturbance-aware MFMS checkpoint improves combined-disturbance robustness, especially success rate, step count, and yaw.
It is not solved yet: mean final XY is still near/above the strict insertion tolerance, so failures remain mostly XY/crop-shift related.
Next useful run should emphasize XY recovery under mask shift/combined disturbances, or train on a mixed profile curriculum rather than only the combined profile.
```

Artifacts:

```text
peg-in-hole-sfnrtifacts
obustness_eval_20260703_disturbance_mfmspeg-in-hole-sfnrtifacts
obustness_eval_20260703_combined_old_mfms_e20peg-in-hole-sfnrtifacts
obustness_eval_20260703_combined_disturbance_mfms_e20```


## 2026-07-03 MFMS Robustness Iteration Result

After the first disturbance-aware MFMS result, several additional MFMS variants were trained/evaluated because the runs were short enough to iterate locally.

Implemented training improvement:

```text
scripts	rain_mfms.py --clean-target
sfn	raining	rain_mfms.py target_vsn support
```

Purpose:

```text
Allow MFMS to receive disturbed visual-history input while optionally copying a clean SFMS teacher target.
This was tested, but it did not improve robustness.
```

Trained/evaluated candidates:

```text
models\mfms_combined_clean_target_teacher_train_seen_4096_e30.pt  -> worse, 22.5% on main combined e20 check
models\mfms_mask_shift_teacher_train_seen_4096_e30.pt             -> helped mask-shift, poor combined generalization
models\mfms_combined_teacher_train_seen_4096_e30_h8.pt            -> worse than h4/h2
models\mfms_combined_teacher_train_seen_8192_e60_h4.pt            -> lower success despite lower imitation loss
models\mfms_combined_teacher_train_seen_4096_e30_h2.pt            -> best current robust candidate
models\mfms_robust.pt                                             -> alias/copy of the h2 best candidate
```

Three-seed combined-disturbance strict insertion comparison:

```text
seeds = 1900, 2000, 2100
episodes = 20 per held-out shape / 40 per seed
task = insertion
mask_source = predicted
profile = combined

Strict-XY SFMS:
  mean_success_rate = 0.542
  mean_steps = 14.80
  mean_final_xy_error_mm = 0.877
  mean_final_yaw_error_deg = 1.556

Old clean-trained MFMS:
  mean_success_rate = 0.558
  mean_steps = 15.27
  mean_final_xy_error_mm = 0.895
  mean_final_yaw_error_deg = 1.808

Combined-disturbance MFMS h4:
  mean_success_rate = 0.533
  mean_steps = 15.07
  mean_final_xy_error_mm = 0.946
  mean_final_yaw_error_deg = 1.505

Combined-disturbance MFMS h2:
  mean_success_rate = 0.592
  mean_steps = 15.24
  mean_final_xy_error_mm = 0.856
  mean_final_yaw_error_deg = 1.512
```

Interpretation:

```text
Best current robustness checkpoint: models\mfms_robust.pt
It is a modest but real robustness improvement, not a solved result.
MFMS history helps a little under stacked disturbance, but final XY is still too high for reliable strict insertion.
More imitation alone is not enough; the next useful step is likely XY-heavy RL under disturbance or oracle/true-pose teacher targets for disturbed inputs.
```

Report artifact:

```text
peg-in-hole-sfnrtifacts
obustness_eval_20260703_iteration_summary\REPORT.md
```

Validation after code changes:

```text
Full pytest: 38 passed in 20.39s
```


## 2026-07-03 Disturbance-Aware RL Fine-Tuning Result

Implemented:

- `scripts/train_sfms.py --robustness-profile` for disturbed SFMS A2C fine-tuning.
- MFMS recurrent A2C fine-tuning in `sfn/training/train_mfms.py`.
- `scripts/train_mfms.py` RL mode using `--init-policy`, `--updates`, `--rollout-steps`, `--best-out`, and `--robustness-profile`.

Key checkpoints produced:

- `models\sfms_combined_robust_a2c_u300_s64.pt`
- `models\sfms_combined_robust_a2c_best.pt`
- `models\mfms_combined_robust_a2c_h2_u300_s64.pt`
- `models\mfms_combined_robust_a2c_h2_best.pt`

Main result:

- Current selected robust imitation checkpoint remains `models\mfms_robust.pt`.
- MFMS RL final improved mean XY slightly across seeds 1900/2000/2100: 0.856 mm -> 0.834 mm.
- MFMS RL success dropped: 59.2% -> 56.7%.
- Therefore, do not promote the RL checkpoint yet.

Report artifact:

- `peg-in-hole-sfnrtifacts
obust_rl_20260703\REPORT.md`

Validation:

- Full pytest: 38 passed in 22.51s.

Recommended next step:

- Do not repeat the same imitation/A2C loop.
- Add temporal filtering or confidence-aware smoothing over VSN probability states, or train from oracle/true-residual targets while keeping disturbed visual inputs.
- Finish Panda native camera/body-ID segmentation before claiming full Panda visual-loop validation.


## 2026-07-03 Robust VSN Ensemble Result

Implemented:

- `EnsembleVirtualSensorNetwork` in `sfn/evaluation/disturbance.py`.
- `TemporalSmoothedVirtualSensorNetwork` in `sfn/evaluation/disturbance.py`.
- `scripts/evaluate_robustness.py --ensemble-samples`.
- `scripts/evaluate_robustness.py --temporal-alpha`.
- Tests proving wrappers call the wrapped VSN instead of bypassing it.

Important correction:

- Temporal smoothing was initially checked, but after ensuring the wrapper preserved the disturbed VSN path, temporal smoothing alone did not beat the baseline.
- VSN probability ensembling did beat the baseline clearly.

Main selected robustness setting:

- checkpoint: `models\mfms_robust.pt`
- inference wrapper: VSN probability ensemble with `--ensemble-samples 5`

Three-seed combined-disturbance strict insertion result:

```text
seeds = 1900, 2000, 2100
episodes = 20 per held-out shape / 40 per seed
task = insertion
mask_source = predicted
profile = combined

Previous robust MFMS:
  mean_success_rate = 0.592
  mean_steps = 15.24
  mean_final_xy_error_mm = 0.856
  mean_final_yaw_error_deg = 1.512

Robust MFMS + VSN ensemble=5:
  mean_success_rate = 0.900
  mean_steps = 9.01
  mean_final_xy_error_mm = 0.473
  mean_final_yaw_error_deg = 0.740
```

This crosses the earlier robustness target:

```text
success >= 75%
mean XY <= 0.70 mm
```

Report artifact:

- `peg-in-hole-sfnrtifacts\ensemble_eval_20260703\REPORT.md`

Validation:

- Full pytest: 41 passed in 21.40s.

Next recommended work:

- Wire ensemble inference into the normal controller/evaluate path if needed outside robustness evaluation.
- Measure/consider latency because ensemble=5 means five perception passes per control step.
- Continue Panda native camera/body-ID segmentation work before claiming full Panda visual-loop validation.


## 2026-07-03 Normal Eval Ensemble + Panda Native Camera Update

Normal evaluation path updated:

- `scripts/evaluate.py` now supports:
  - `--ensemble-samples`
  - `--robustness-profile`
  - `--temporal-alpha`
- This makes the best ensemble setting usable outside `evaluate_robustness.py`.

Panda controller path updated:

- `scripts/panda_evaluate_controller.py` now supports:
  - `--ensemble-samples`
  - `--native-camera`

Panda native camera/body-ID segmentation added:

- `PandaConfig(native_camera=True)` switches Panda observations to PyBullet camera output.
- `PandaScene.render_camera()` renders RGB plus body-ID segmentation mask.
- Mask labels are background=0, peg=1, base/hole body=2.
- Added `scripts/panda_validate_native_camera.py`.
- Added test coverage for native Panda camera mask labels.

Final normal-path evaluation with `models\mfms_robust.pt`, `--ensemble-samples 5`:

```text
Clean strict insertion, test_unseen, predicted masks:
  episodes = 200
  success_rate = 1.000
  mean_steps = 5.07
  mean_final_xy_error_mm = 0.349
  mean_final_yaw_error_deg = 0.543

Combined disturbance, seeds 1900/2000/2100, 40 episodes each:
  mean_success_rate = 0.858
  mean_steps = 9.88
  mean_final_xy_error_mm = 0.490
  mean_final_yaw_error_deg = 0.769
```

Panda native camera smoke:

```text
scripts\panda_validate_native_camera.py:
  success = true
  mask contains peg and base labels

Panda oracle with --native-camera:
  success_rate = 1.0 on smoke

SFMS learned controller with native body-ID masks:
  success_rate = 0.0 on smoke
  reason: expected camera/crop geometry mismatch between native Panda render and synthetic VSN training distribution
```

Updated combined report:

- `peg-in-hole-sfnrtifacts\combined_progress_report_20260703\REPORT.md`

Next Panda milestone:

- Calibrate native camera crop/scale to the VSN training distribution or retrain VSN on native Panda-rendered body-ID masks.


## 2026-07-03 Panda Native Camera Calibration/VSN Diagnosis

Further Panda native-camera work completed:

- Changed `PandaConfig.camera_eye_offset_m` default to `(0.0, 0.0, 0.10)`.
  - Previous 0.05 m view made the peg dominate the crop.
  - 0.10 m produces a more useful native body-ID mask scale.
- Added `scripts/collect_panda_native_dataset.py`.
  - Collects Panda PyBullet RGB/body-ID masks into the same NPZ/manifest format as the synthetic dataset.
  - Supports `--camera-z`.
- Added `scripts/panda_analyze_native_masks.py`.
  - Fits simple centroid/moment diagnostics to check whether native masks contain enough pose information.

Collected probe datasets:

```text
artifacts\panda_native_vsn_20260703\z010_train_probe
artifacts\panda_native_vsn_20260703\z010_val_probe
```

Observability result at native camera z=0.10 m:

```text
XY from body-ID mask centroids:
  validation mean error = 0.427 mm
  validation p90 error  = 0.825 mm

Yaw from peg-mask principal axis:
  validation mean error = 4.34 deg
  validation p90 error  = 9.22 deg
  within 2 deg          = 37.7%
```

Interpretation:

- Native Panda body-ID masks contain usable XY information after camera scale calibration.
- Native body-ID masks do not yet contain reliable yaw information for strict insertion.
- This explains why the learned controller failed on native Panda masks: the VSN training distribution and yaw signal do not match.

Training note:

- A first native Panda position/orientation training attempt was started.
- It is too slow on CPU with the current high-resolution perception architecture, and the current `OrientationNet` is mostly analytic/logit-scale based rather than a fully learnable CNN.
- Do not spend more time on the same training command until the orientation architecture/camera signal is fixed.

Useful commands if a longer native-position training run is still wanted:

```powershell
..\.venv\Scripts\python.exe scripts	rain_position.py `
  --dataset data\panda_native_train_seen `
  --val-dataset data\panda_native_validation_unseen `
  --out models\panda_native_position.pt `
  --epochs 40 `
  --batch-size 64 `
  --lr 0.001 `
  --base-channels 16 `
  --seed 3700 `
  --device auto `
  --metric exact_cell_accuracy `
  --patience 10
```

But the better next step is to implement a Panda-native XY bridge and redesign/retrain orientation for native yaw.

Validation:

```text
scripts\panda_validate_native_camera.py: success
Full pytest: 41 passed in 22.62s
```


## 2026-07-03 Panda Native Geometric VSN Bridge

Added:

- `sfn\panda
ative_vsn.py`
- `PandaBodyIdGeometricVSN`
- `scripts\panda_evaluate_controller.py --native-geometric-vsn`
- Test coverage for native geometric VSN output shape/state.

Smoke result:

```text
..\.venv\Scripts\python.exe scripts\panda_evaluate_controller.py `
  --method sfms `
  --policy models\sfms_strict_xy_anchor1_best.pt `
  --episodes 1 `
  --split validation_unseen `
  --out artifacts\panda_native_vsn_20260703\sfms_native_geometric_smoke `
  --native-camera `
  --native-geometric-vsn `
  --seed 4000
```

Output:

```text
success_rate = 0.0
mean_final_xy_error_mm = 0.547
mean_final_yaw_error_deg = 7.830
```

Interpretation:

- Native body-ID masks can now drive XY close to strict tolerance.
- Native yaw remains the blocker.
- Do not spend more compute on old orientation training as-is; current `OrientationNet` is not a proper native-yaw CNN.
- Next useful work is yaw observability: camera angle/render cue, asymmetric body-ID signal, or learned native orientation architecture.

Validation:

```text
Full pytest: 42 passed in 22.81s
```

# 2026-07-13 Final Software Release — Supersedes Earlier Notes

The supported software implementation and release audit are complete. Earlier notes below are retained as development history and must not be used as current status.

Authoritative report set:

- `peg-in-hole-sfn/artifacts/software_completion_20260713/FINAL_SOFTWARE_REPORT.md`
- `peg-in-hole-sfn/artifacts/software_completion_20260713/IMPLEMENTATION_STATUS.md`
- `peg-in-hole-sfn/artifacts/software_completion_20260713/COMPLETION_CERTIFICATE.md`
- `peg-in-hole-sfn/artifacts/software_completion_20260713/final_benchmark_release/`

Current headline results:

- Mesh predicted-mask, three seeds: SFSS 120/120 at 0.228 mm; SFMS 107/120 at 0.278 mm; MFMS 120/120 at 0.242 mm.
- Panda predicted-mask, three seeds: SFSS 29/30 at 0.252 mm; SFMS 30/30 at 0.193 mm; MFMS 30/30 at 0.333 mm.
- Final Panda matrix predicted masks: SFSS 17/20; SFMS 19/20; MFMS 19/20.
- Severe camera disturbance: SFSS 22/40; SFMS 23/40; MFMS 24/40. Robustness is not solved.
- MFMS history 1/2/4/8 under burst-5 corruption: 78/80, 69/80, 59/80, 48/80. Temporal advantage was not demonstrated.
- Full one-frame mesh predicted-mask cascade remains weaker than closed loop: 1.607 mm mean X-Y, 0.622 mm median, 2.224° yaw, 9.01% invalid masks.

Panda status:

- Native camera, explicit downward attachment, measured motion, IK, dynamic motor execution, contact, depth, jams/timeouts, and detailed telemetry are implemented.
- All-shape exact dynamic insertion is 16/16; repeated oracle alignment is 80/80.
- The full ±10 mm camera workspace is unclipped, but exact X-Y alignment physically hides the seam in 36/432 sweep frames. Universal single-frame yaw observability is not claimed.

Reproducibility:

- Final release checkpoints have additive provenance metadata with unchanged model-state hashes.
- Schema-v2 datasets, cards, checkpoint registry, environment lock, source manifest, deterministic source ZIP, full benchmark driver, CPU quality gate, GPU parity smoke, raw traces, statistics, plots, and videos are present.
- The historical Git working tree is dirty; the deterministic source snapshot/archive defines the portable release source.

Remaining work is hardware-only commissioning and optional research improvement: physical metrology, camera/tool calibration, real-frame collection and segmentation adaptation, guarded Franka integration, force limits, and matched simulation/hardware trials. Do not report any present number as hardware performance.

---
