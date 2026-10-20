# Jetson hardware cookbook

This checklist ends at a motors-disabled dry-run. Motored navigation has a separate acceptance gate.

## 1. Preconditions

- [ ] The operator has physical access, a clear test area, and a tested emergency stop.
- [ ] The current lab-approved network path reaches both compute and robot endpoints.
- [ ] JetPack, ROS 2, and the NVIDIA PyTorch build are installed and compatible.
- [ ] An external RGB-D camera is physically mounted and enumerates on the Jetson.
- [ ] NanoSAM engine files exist if the NanoSAM profile will be used.
- [ ] The base workspace builds. No claim is made yet that its navigation path is hardware-ready.

## 2. Build and static checks

```bash
cd <repo>
source /opt/ros/humble/setup.bash
source <base-workspace>/install/setup.bash

python3 -m pip install -r requirements.txt
cd ros2_ws
colcon build --symlink-install --packages-up-to go2_semantic_bringup
source install/setup.bash
ros2 launch go2_semantic_bringup semantic_nav.launch.py --show-args
```

Do not reinstall PyTorch unless following the NVIDIA instructions for the exact JetPack release.

### Offline detector dependencies (no internet on the robot)

Verified on JetPack 6 (L4T R36.4.7, CUDA 12.6, cuDNN 9.3), Python 3.10. A
PyPI torch wheel for aarch64 imports but reports `cuda False` on Jetson; use
the Jetson AI Lab build. Everything goes into one private folder that only the
detector's environment puts on `PYTHONPATH`, so a system or user-site torch is
left alone.

On a PC with internet:

```bash
IDX=https://pypi.jetson-ai-lab.io/jp6/cu126/+simple
pip download --no-deps --only-binary=:all: --platform linux_aarch64 --python-version 310 \
    --implementation cp --index-url $IDX torch==2.8.0 torchvision==0.23.0 -d wheels
pip download --no-deps --only-binary=:all: --platform manylinux2014_aarch64 --python-version 310 \
    --index-url $IDX nvidia-cusparselt-cu12==0.7.1 -d wheels
pip download --no-deps --only-binary=:all: --platform manylinux2014_aarch64 --python-version 310 \
    open_clip_torch timm ftfy wcwidth -d wheels
pip wheel --no-deps -w wheels git+https://github.com/ultralytics/CLIP.git \
    git+https://github.com/ChaoningZhang/MobileSAM.git
```

Copy `wheels/` and the weights to the robot computer: `yolov8s-worldv2.pt` to
`~/.cache/ultralytics/`, `mobile_sam.pt` to `~/.cache/mobile_sam/`, OpenAI
CLIP `ViT-B-32.pt` (YOLO-World encodes its prompts with it) to `~/.cache/clip/`,
and the Hugging Face cache folder `models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K`
to `~/.cache/huggingface/hub/`. Then on the robot computer:

```bash
T=~/gsn_lab/pydeps_ml
pip3 install --no-deps --no-index --target $T wheels/*.whl
export PYTHONPATH=$T:$PYTHONPATH LD_LIBRARY_PATH=$T/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1 YOLO_OFFLINE=1
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.8.0 True
```

Measured on the Orin NX (MAXN) on one live 1280x720 frame with 11 objects:
YOLO-World v2-s 34 ms, MobileSAM 573 ms, OpenCLIP ViT-B/16 511 ms, about
1.1 s per detection cycle; masks and embeddings dominate.

## 3. Discover live interfaces

```bash
ros2 topic list -t
ros2 node list
ros2 topic hz <color-image-topic>
ros2 topic hz <aligned-depth-topic>
ros2 topic echo <color-camera-info-topic> --once
ros2 run tf2_ros tf2_echo map <color-optical-frame>
```

Pass criteria:

- color and aligned depth are sustained, not one-shot
- color, depth, and camera info use compatible dimensions and the same optical frame
- `map` to camera TF is current and continuous
- there are no competing publishers on the same semantic output topics

## 4. Start perception with motors disabled

```bash
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
  enable_grounding:=false \
  use_rviz:=false
```

Confirm `/semantic/detections` and `/semantic/scene_graph` are sustained and inspect their frame IDs, positions, labels, latency fields, and memory use.

## 5. Dry-run grounding

Only enable grounding after a fresh global costmap exists in `map`:

```bash
ros2 topic hz /global_costmap/costmap
ros2 action send_goal /semantic/ground_and_navigate \
  go2_semantic_msgs/action/GroundAndNavigate \
  "{text_query: 'go near the chair', stand_off_m: 0.9, dry_run: true}" \
  --feedback
```

Pass criteria:

- stale or missing scene graphs are rejected
- stale, missing, or wrong-frame costmaps are rejected
- unknown and ambiguous targets fail without a goal
- a valid request returns a `map`-frame pose
- `/goal_pose` remains silent because `allow_goal_publication` is false

Capture the session with `scripts/record_demo_bag.sh` and `scripts/diagnose.sh`.

## 6. Thermal evidence

Run the target workload long enough to reach thermal steady state. Record power mode, clocks, temperatures, throttling flags, memory use, and sustained detector rate. Do not publish performance numbers until they come from this run.

## 7. Separate motion acceptance gate

Do not set `allow_goal_publication:=true` until the base stack independently proves:

- continuous odometry and `map -> odom -> base_link` TF
- localization or mapping appropriate to the environment
- live obstacle data feeding the global and local costmaps
- a consumer for `/goal_pose` with observable success, abort, and cancel behavior
- operator stop, communications-loss stop, and stale-data stop behavior
- low-speed bounded tests before semantic goals are introduced

Once those pass, enable publication explicitly for the supervised motion session. The interlock is intentionally off by default on every launch.
