# SFN Peg-in-Hole Handover Notes

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
