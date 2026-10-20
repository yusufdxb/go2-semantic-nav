# Deployment

The first hardware milestone is a motors-disabled perception and grounding run. This repository is not sufficient by itself for motored navigation.

## Readiness boundary

The semantic overlay requires these external contracts:

- aligned color and depth images plus color camera intrinsics
- TF from the color optical frame to `map`
- a current global costmap in `map` when `use_costmap_gate` is enabled
- for motion only: proven odometry, robot TF, localization or mapping, a `/goal_pose` consumer, and an independently tested stop path

The sibling base stack (GO2-seeing-eye-dog, branch `feat/nav2-deploy`) now
provides these contracts: odometry and LiDAR relayed with robot-clock
correction, `map -> odom` from slam_toolbox, Nav2 serving `NavigateToPose`,
a LiDAR hazard source, and the safety arbiter as the only path to the robot.
They are verified in closed-loop simulation only (see `RESULTS.md`). On the
robot, follow the staged runbook in that repository, `docs/DEPLOYMENT.md`:
preflight, motors untouched, dry-run actuator, then low-speed motion. Keep
`allow_goal_publication:=false` until its stage 3 passes.

One launch brings up both stacks:

```bash
ros2 launch go2_semantic_bringup deploy.launch.py \
  planner:=nav2 localization:=slam_mapping hardware_adapter:=dry_run \
  allow_goal_publication:=false
```

Motion needs all three: `hardware_adapter:=unitree_sport`,
`allow_goal_publication:=true`, and a request with `dry_run: false`.

## Workstation setup

```bash
source /opt/ros/humble/setup.bash
source <base-workspace>/install/setup.bash

cd <repo>
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

cd ros2_ws
colcon build --symlink-install --packages-up-to go2_semantic_bringup
source install/setup.bash
```

## Jetson setup

Use the NVIDIA PyTorch build matched to the installed JetPack release. Do not install a generic desktop CUDA wheel. Verify the platform installation before adding project dependencies:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Then clone and build the repository directly:

```bash
git clone https://github.com/yusufdxb/go2-semantic-nav.git
cd go2-semantic-nav
python3 -m pip install "numpy<2.0" ultralytics open_clip_torch networkx pyyaml

source /opt/ros/humble/setup.bash
source <base-workspace>/install/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to go2_semantic_bringup
source install/setup.bash
```

Preload model weights while network access is available:

```bash
cd <repo>
python3 scripts/prefetch_models.py
```

## Camera discovery

Topic names differ between RealSense driver versions. Discover the live names instead of assuming the defaults:

```bash
ros2 topic list -t | grep -E 'camera|image|camera_info'
ros2 topic hz <color-image-topic>
ros2 topic hz <aligned-depth-topic>
ros2 topic echo <color-camera-info-topic> --once
```

Depth must already be registered to the color optical frame. The detector rejects mismatched frames or image dimensions by default.

## Backend reality

The standard YOLO-World and OpenCLIP backends currently run through their PyTorch implementations. The export scripts produce benchmarking artifacts, but those engines are not consumed by the runtime backends. Do not claim TensorRT acceleration for them.

NanoSAM is the available TensorRT-native segmentation path. It requires both engine files before startup:

```bash
export NANOSAM_ENCODER_ENGINE=<path>/resnet18_image_encoder.engine
export NANOSAM_DECODER_ENGINE=<path>/mobile_sam_mask_decoder.engine
test -r "$NANOSAM_ENCODER_ENGINE"
test -r "$NANOSAM_DECODER_ENGINE"
```

## Motors-disabled first launch

Use one installed scene profile for all three nodes, remap the discovered camera topics, and leave goal publication disabled:

```bash
source /opt/ros/humble/setup.bash
source <base-workspace>/install/setup.bash
source <repo>/ros2_ws/install/setup.bash

PROFILE="$(ros2 pkg prefix go2_semantic_bringup)/share/go2_semantic_bringup/config/scene_profiles/jetson_tier_a.yaml"

ros2 launch go2_semantic_bringup semantic_nav.launch.py \
  detector_params:="$PROFILE" \
  scene_graph_params:="$PROFILE" \
  grounding_params:="$PROFILE" \
  image_topic:=<color-image-topic> \
  depth_topic:=<aligned-depth-topic> \
  camera_info_topic:=<color-camera-info-topic> \
  require_aligned_depth:=true \
  allow_goal_publication:=false \
  detection_rate_hz:=3.0
```

If the validated base stack is not publishing a current global costmap, disable grounding for the perception-only portion with `enable_grounding:=false`. Do not bypass the costmap gate to simulate navigation readiness.

## Optional offboard detector

A companion workstation can publish `/semantic/detections` over a lab-approved DDS configuration. Launch the onboard stack with `enable_detector:=false`. Network and DDS settings are deployment-specific and must not be committed to this public repository.

## Disable the overlay

Stop the launch process and source only the base workspace. This repository does not modify the sibling workspace.
