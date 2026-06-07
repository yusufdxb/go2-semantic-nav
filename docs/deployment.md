# Deployment

Two target environments: **dev workstation (`mewtwo`, RTX 5070)** and **robot onboard (Jetson Orin NX 16 GB)**. Configurations, weights, and dependency paths differ; interfaces do not.

## Dev: `mewtwo` (RTX 5070, CUDA 12.8)

### One-time setup
```bash
# Source the sibling seeing-eye-dog workspace first (for go2_msgs + nav2 overlay)
source ~/ros2_ws/install/setup.bash

# Create Python venv for this project's ML deps
cd ~/Projects/personal/go2-semantic-nav
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Build the ROS 2 workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# Smoke test: should show 5 packages
ros2 pkg list | grep go2_
```

### Running with a rosbag
```bash
# Terminal A: sibling stack (sim or real camera driver)
ros2 launch go2_bringup go2_full.launch.py use_sim:=true

# Terminal B: rosbag play (pre-recorded indoor scene)
ros2 bag play ~/data/go2-indoor-sample/ --loop

# Terminal C: this stack
ros2 launch go2_semantic_bringup semantic_nav.launch.py \
    device:=cuda:0 \
    backend:=yolo_world_v2_s \
    encoder:=openclip_vit_b16
```

### First-run model downloads
On first launch the nodes pull weights to `~/.cache/...`:

- YOLO-World: `ultralytics` cache → `~/.config/Ultralytics/`
- MobileSAM: HuggingFace cache → `~/.cache/huggingface/`
- OpenCLIP: `open_clip_torch` cache → `~/.cache/clip/` or `~/.cache/huggingface/`

Pre-seed offline environments by running `python scripts/prefetch_models.py`.

## Robot onboard: Jetson Orin NX 16 GB (JetPack 6.x)

### One-time setup (on the Jetson, via SSH from laptop)
```bash
# Confirm JetPack 6.x and CUDA 12.x
cat /etc/nv_tegra_release
nvcc --version

# ROS 2 Humble from apt (see go2-jetson-setup-guide repo for full path)
# Clone this repo under ~/go2_ws_overlay/src/
mkdir -p ~/go2_ws_overlay/src && cd ~/go2_ws_overlay/src
git clone git@github.com:<owner>/go2-semantic-nav.git
cd ~/go2_ws_overlay && ln -s src/go2-semantic-nav/ros2_ws/src . 2>/dev/null || true

# PyTorch on Jetson: use NVIDIA's wheel, NOT pip default
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.3.0 torchvision==0.18.0

# ML deps: note: skip mobile_sam on Jetson if using NanoSAM
pip install ultralytics open_clip_torch transformers networkx open3d

# sentencepiece: build from source on aarch64
pip install sentencepiece --no-binary=:all:

# Build the overlay
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash       # sibling seeing-eye-dog overlay
cd ~/go2_ws_overlay
colcon build --symlink-install
source install/setup.bash
```

### TensorRT export (offline, once per backend)

```bash
# YOLO-World v2-s → ONNX → TRT FP16
python scripts/export_yolo_world_trt.py \
    --model yolov8s-worldv2.pt \
    --prompts config/prompts_indoor.yaml \
    --out models/yolo_world_v2_s_indoor.engine \
    --fp16

# OpenCLIP image encoder → ONNX → TRT FP16
python scripts/export_openclip_trt.py \
    --model ViT-B-16 \
    --pretrained laion2b_s34b_b88k \
    --out models/openclip_vit_b16.engine \
    --fp16

# MobileSAM → ONNX → TRT FP16 (decoder is single-prompt; encoder batched)
python scripts/export_mobilesam_trt.py \
    --checkpoint models/mobile_sam.pt \
    --out-encoder models/mobilesam_encoder.engine \
    --out-decoder models/mobilesam_decoder.engine
```

**Gotchas captured from research:**
- YOLO-World with dynamic text-token dim: **re-parameterize to fixed vocab before ONNX export** via `model.set_classes(names); model.save()`. Otherwise the TRT engine must be rebuilt on every vocab change.
- NanoSAM pre-built engines target JP5; on JP6 rebuild from ONNX with explicit static shapes: `trtexec --onnx=... --fp16 --shapes=image:1x3x1024x1024`.
- Grounding-DINO TRT needs the `MultiScaleDeformableAttn` plugin which is not in stock L4T TRT 10.x. Not recommended for on-robot use.
- OpenCLIP ViT-L/14 TRT: force FP16 precision constraints to avoid GELU precision fallback: `--precisionConstraints=obey --layerPrecisions=*:fp16`.
- SigLIP text tokenizer requires `sentencepiece` from-source build on aarch64.

### Launching on the robot
```bash
source ~/go2_ws_overlay/install/setup.bash

# Launch with Jetson backend profile
ros2 launch go2_semantic_bringup semantic_nav.launch.py \
    device:=cuda:0 \
    backend:=yolo_world_v2_s \
    segmenter:=nano_sam \
    encoder:=mobileclip_s2 \
    use_tensorrt:=true \
    detection_rate_hz:=3.0
```

### Power mode
Set the Jetson to 25 W sustained before benchmarking or demos:
```bash
sudo nvpmodel -m 0      # MAXN (25 W on Orin NX 16 GB)
sudo jetson_clocks
```

### Known constraints
- 25 W sustained budget → detector + SAM + CLIP + scene graph must fit in ≤200 ms/frame at 3-5 Hz with Nav2 headroom.
- OpenCV + cv_bridge pin `numpy<2.0`. Do not let `pip install` bump it.
- `mobile_sam` pip package is x86-friendly; on Jetson prefer direct checkpoint + NanoSAM TRT engines.

## Offboard companion (optional, Tier C)

If a laptop (mewtwo) is co-located on the GO2's `192.168.123.0/24` LAN, heavier models can run there and publish to the robot over DDS:

```bash
# On mewtwo: acts as a companion publisher
export ROS_DOMAIN_ID=7              # match robot's domain
ros2 launch go2_semantic_bringup detector_offboard.launch.py \
    backend:=grounding_dino_tiny \
    encoder:=clip_vit_h14
```

On the robot, disable the onboard detector and subscribe to the companion's `/semantic/detections`:
```bash
ros2 launch go2_semantic_bringup semantic_nav.launch.py \
    use_offboard_detector:=true
```

This shifts detection compute off the Jetson while keeping scene-graph state and grounding onboard where it gates Nav2 goals.

## Uninstall / disable

To run the seeing-eye-dog stack without this overlay:
```bash
# Just do not source this overlay
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch go2_bringup go2_full.launch.py
```

No residual state is left behind because this overlay does not modify any file in the sibling workspace.
