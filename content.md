rough notes on this SFN / peg-in-hole repo

this is my current understanding of the paper + code. not a perfect official doc, just notes so i can explain the project.

main idea

the project is about doing peg-in-hole using vision. instead of directly taking an RGB image and predicting robot action, they first convert the camera image into a simpler mask. the mask has classes like background, peg, and seam/hole gap. then the rest of the system only looks at this mask and tries to align the peg with the hole.

the important idea is "fill the seam". if peg is not aligned with hole, some gap or seam is visible. if robot moves correctly, the seam becomes smaller / filled. so the task becomes: look at peg + seam mask, estimate how to move in x/y and rotate in yaw, then repeat.

overall flow  

camera image comes in
segmentation network makes a mask
position network predicts x/y correction
orientation network predicts yaw correction
robot applies the correction
camera sees again
repeat until aligned
then actual insertion / pushing is done with force or contact control

in simulation the repo often skips the segmentation network because the simulator can directly give the correct mask as obs['gt']. so a lot of code uses gt mask directly. in real world we would need the segmentation network because there is no automatic perfect mask from camera.

important folders/files
 

peg-in-hole-sfn/gymEnv/envs/peg_in_hole_v11.py
this is the main pybullet simulation environment.
it loads the peg and hole models, renders images, creates masks, and returns observations.
the observation is roughly:
img = RGB camera image
gt = ground truth segmentation mask
dxy = real x/y error
dyaw = real yaw error

this file also has step(action). in test mode the action changes the peg pose. in train mode it seems to randomly sample perturbations, which is useful for collecting supervised data for the position/orientation modules.

peg-in-hole-sfn/gymEnv/envs/complex/
this is where the actual peg/hole models are.
each shape has folders/files like peg.urdf, peg.obj, base.urdf, base.obj, mask.obj.
examples are square-triangle, square-square, square-pentagon, square-concave1, square-fillet1 etc.
so the pybullet model assets are already in the repo.

segmentation module

main file:
peg-in-hole-sfn/algos/pytorch/fcn/seg_ur5_real.py

network file:
peg-in-hole-sfn/algos/pytorch/fcn/unet.py

this trains a U-Net with 3 input channels and 3 output classes:
UNet(3, 3)

input is normal RGB image.
output is a segmentation mask with 3 classes.
roughly:
0 background
1 peg
2 seam / hole visible region

the correct mask used for training is called ground truth mask. in simulation this is easy because simulator knows everything. in real world the paper says they render masks automatically using robot pose + 3d model, after one human demonstration.

important thing: the repo imports UR5Dataset from dataloader, but the dataloader.py source does not seem to be present, only pycache maybe. so real-world segmentation training is only partially usable unless dataloader is recovered/rebuilt.

what is U-Net

U-Net is a neural network used for image-to-image type tasks. it takes an image or mask and outputs another image/grid. it first compresses the image to understand features, then expands it back to spatial output.

here U-Net is used in different ways:
RGB image to segmentation mask
segmentation mask to position heatmap
mask to feature map for orientation matching

position alignment module

main file:
peg-in-hole-sfn/algos/pytorch/fcn/position_11.py

network:
peg-in-hole-sfn/algos/pytorch/fcn/unet_11.py

helper:
peg-in-hole-sfn/utils/utils.py

this module answers:
how much should peg move in x and y?

input is segmentation mask, not RGB image.
output is a 21x21 heatmap.

the 21x21 heatmap  basically represents possible x/y correction commands. it is not image pixels. it is movement/error space.

why 21x21:
the paper/environment assumes x and y error is around -10 mm to +10 mm.
if we use 1 mm resolution, then possible values are:
-10, -9, -8, ..., 0, ..., 8, 9, 10
that is 21 possible values for x and 21 possible values for y.
so 21 x 21 = 441 possible x/y correction bins.

inside the heatmap:
each cell has a score for "this is the correct x/y correction".
the brightest/highest cell is selected.

center cell means zero correction:
row 10, col 10 means dx=0, dy=0
other cells mean move by some millimeters.

during training, simulator knows true dxy. utils.get_position_gt converts true dxy to a one-hot heatmap. one-hot means all zeros except one cell is 1. that one cell is the correct answer.

roughly:
dxy is in meters
multiply by 1000 to get mm
add 10 so -10mm becomes index 0, 0mm becomes index 10, +10mm becomes index 20

then the position U-Net learns:
given this mask, put the high value in the right cell of the 21x21 heatmap.

orientation alignment module

main file:
peg-in-hole-sfn/algos/pytorch/fcn/pose_8.py

network:
peg-in-hole-sfn/algos/pytorch/fcn/unet.py

this module answers:
how much should peg rotate in yaw?

it works differently from position. it does not output a 21x21 grid. it compares possible rotations.

possible rotations are:
-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 degrees

so 11 candidates total.
this is because yaw error range is about -10 to +10 degrees and resolution is 2 degrees.

logic:
take segmentation mask
make peg-only mask
make seam/hole-only mask
rotate the seam mask by each candidate angle
send peg mask and rotated seam masks through same U-Net
compare their feature maps using distance
smallest distance means best match
that angle is predicted yaw correction

this is like shape matching. instead of directly predicting "angle = 6 degrees", it asks "which rotated seam best matches the peg?"

this should generalize better to unseen shapes because it learns matching/alignment, not memorizing a specific shape.

training orientation:
the correct rotation is the positive pair. wrong rotations are negative pairs.
the contrastive loss tries to make correct peg-seam pair close in feature space, and wrong rotations far away.

training scripts

peg-in-hole-sfn/train_position_11.py
entry point for training position module.
it creates gym envs using peg-in-hole-v11, wraps with SubprocVecEnv, then calls position.train_position.

peg-in-hole-sfn/train_pose_8.py
entry point for training orientation/yaw module.
also creates gym envs and calls pose.train_pose.

in these training files, there are many commented env shapes. probably authors changed experiments by commenting/uncommenting shapes manually.

inference/testing scripts

peg-in-hole-sfn/test_pose_position_gui_11.py
this is the most useful script to understand the closed loop. it loads pose_model and position_model, creates simulation env, then repeatedly:
gets dx/dy from position model
gets dyaw from orientation model
applies correction using env.step
checks if alignment is good enough

it mostly uses obs['gt'] as mask, not learned segmentation. there is commented code showing where seg_model would go.

peg-in-hole-sfn/test.py
another test/experiment file. has functions get_dxy and get_dyaw too. seems messy and partly exploratory.

peg-in-hole-sfn/test_pose_gui_8.py
tests only orientation/yaw, but it references older env name v8 which is not present as source in this checkout.

RL parts

paper has three versions:
SFSS = single frame single step
SFMS = single frame multi step with RL
MFMS = multi frame multi step with RNN/LSTM + RL

my broad understanding:
SFSS just takes current mask, predicts dx/dy/dyaw, moves, and repeats.
SFMS uses RL to choose better actions over multiple future steps instead of greedily taking current best correction.
MFMS also uses history of multiple frames, so it can handle cases where current view is ambiguous or seam is hidden.

repo has RL folders:
peg-in-hole-sfn/algos/pytorch/a2c/
peg-in-hole-sfn/algos/pytorch/a2c_9/
peg-in-hole-sfn/algos/pytorch/a2c_rnn/
peg-in-hole-sfn/algos/pytorch/a2c_fusion/
peg-in-hole-sfn/algos/pytorch/a2c_rnn_encoder/
also ppo, sac, td3, ddpg, etc.

main A2C file:
peg-in-hole-sfn/algos/pytorch/a2c/a2c.py

but I do not clearly see a complete top-level training script that wires the paper's position/orientation heatmaps into RL exactly as described. also peg_in_hole_v11.step currently returns reward 0 and done False, so it does not look like a complete RL environment in the present source.

simulation details

environment:
peg-in-hole-sfn/gymEnv/envs/peg_in_hole_v11.py

assets:
peg-in-hole-sfn/gymEnv/envs/complex/

the env uses pybullet for robot/physics and pyrender/trimesh for masks. it loads panda robot/peg and base hole models. the rendered output is cropped to about 200x250 for mask/image use.

important variables returned:
img: camera image
gt: segmentation mask
dxy: true x/y displacement
dyaw: true yaw displacement

data flow in code

for training position:
env gives obs
obs['gt'] is used as input mask
obs['dxy'] is converted to 21x21 one-hot heatmap
position model learns mask to heatmap

for training orientation:
env gives obs
obs['gt'] is split into peg mask and seam mask
seam mask is rotated by candidate angles
model learns correct rotation should match peg features better than wrong rotations

for inference:
env gives obs
position model predicts heatmap
argmax heatmap gives dx/dy
orientation model compares rotations
best rotation gives dyaw
env.step applies correction
repeat

real-world part

paper says real-world transfer only needs retraining segmentation module. reason is position/orientation modules operate on masks, not raw RGB. if segmentation works on real image, the rest can stay simulation-trained.

repo has seg_ur5_real.py for this idea, but not full robot deployment. I don't see actual UR5 live control/camera/force-control deployment code in source.

simple way to explain whole project

this is not really "robot sees RGB and magically acts".
it is more structured:
first simplify image into peg/seam mask
then estimate translation from heatmap
then estimate rotation by matching rotated seam masks
then run it closed loop

the seam mask is the key representation. it makes different shapes look like the same kind of alignment problem.

The segmentation mask is the main part of the whole algorithm. Once you get the mask, afterwars its just a simple robot loop control rather than any specialized hyper-network. 
_____


Current workdone:

getting the pybullet simulation running and buiding the basic environment for running it. 

cmd used:
cd peg-in-hole-sfn
..\.venv\Scripts\python simple_closed_loop_sim.py --episodes 3 --max_steps 10 --peg_type square-concave1 --seed 1

..\.venv\Scripts\python demo_closed_loop_gui.py --peg_type square-concave1 --dx_mm 8 --dy_mm -6 --dyaw_deg 8 --pause 1.0 --hold_seconds 60