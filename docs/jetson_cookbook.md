# Jetson deployment cookbook

Hands-on checklist for bringing go2-semantic-nav up on a Unitree GO2 with an
onboard Jetson Orin NX 16 GB. Run commands from SSH (your laptop) unless marked
`[Jetson]`.

## 0. Preconditions

- [ ] Jetson is flashed with JetPack 6.x per `go2-jetson-setup-guide`.
- [ ] Jetson has a static `192.168.123.15/24` and pings `192.168.123.161`.
- [ ] `GO2-seeing-eye-dog` is cloned at `~/ros2_ws/src/GO2-seeing-eye-dog` and `colcon build` succeeded for it.
- [ ] RealSense driver is running and publishing `/camera/color/image_raw` at ≥15 Hz.
- [ ] Laptop is sharing internet to the Jetson (follow `docs/05-internet-sharing.md` from the setup guide).

## 1. Install deps on the Jetson

```bash
# [Jetson]
sudo nvpmodel -m 0      # MAXN, 25 W sustained
sudo jetson_clocks

# NumPy first (pin for cv_bridge compat)
pip install "numpy<2.0"

# ML stack: do NOT reinstall torch; the JetPack wheel is already present
pip install ultralytics open_clip_torch networkx opencv-python pyyaml
pip install --no-binary=:all: sentencepiece
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

For NanoSAM (preferred segmenter on Orin NX):

```bash
# [Jetson]
cd ~ && git clone --depth 1 https://github.com/NVIDIA-AI-IOT/nanosam.git
cd nanosam && pip install -e .
# Follow nanosam/README.md to build the two TRT engines:
# - resnet18_image_encoder.engine
# - mobile_sam_mask_decoder.engine
# Copy both to ~/nanosam/data/
```

## 2. Clone and build go2-semantic-nav

```bash
# [Jetson]
mkdir -p ~/go2_ws_overlay/src
cd ~/go2_ws_overlay/src
git clone https://github.com/<owner>/go2-semantic-nav.git .
cd ~/go2_ws_overlay
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash   # sibling seeing-eye-dog overlay
colcon build --symlink-install --packages-up-to go2_semantic_bringup
source install/setup.bash
```

## 3. Prefetch model weights (while still on laptop-shared internet)

```bash
# [Jetson]
python3 scripts/prefetch_models.py --backends yolo_world_v2_s,mobile_sam,openclip_vit_b16
```

## 4. Export TensorRT engines (one-time per model × device)

```bash
# [Jetson]
# YOLO-World with our indoor prompt set baked in (fixed-vocab, faster)
python3 scripts/export_yolo_world_trt.py \
    --model yolov8s-worldv2.pt \
    --prompts ros2_ws/src/go2_open_vocab_detector/config/prompts_indoor.yaml \
    --out-onnx models/yolo_world_v2_s.onnx \
    --out-engine models/yolo_world_v2_s.engine \
    --fp16

# OpenCLIP image encoder
python3 scripts/export_openclip_trt.py \
    --model ViT-B-16 --pretrained laion2b_s34b_b88k \
    --out-onnx models/openclip_vit_b16_image.onnx \
    --out-engine models/openclip_vit_b16_image.engine \
    --fp16 --obey-precision
```

## 5. Launch the overlay

```bash
# [Jetson] with seeing-eye-dog bringup already running in another terminal
source ~/go2_ws_overlay/install/setup.bash

ros2 launch go2_semantic_bringup semantic_nav.launch.py \
    device:=cuda:0 \
    backend:=yolo_world_v2_s \
    segmenter:=nano_sam \
    encoder:=openclip_vit_b16 \
    detection_rate_hz:=3.0 \
    publish_rate_hz:=2.0
```

## 6. Verify in RViz on the laptop

```bash
# [Laptop]
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=<match-robot>
ros2 topic hz /semantic/detections      # ~3 Hz
ros2 topic hz /semantic/scene_graph     # ~2 Hz
rviz2 -d $(ros2 pkg prefix go2_semantic_bringup)/share/go2_semantic_bringup/rviz/semantic_nav.rviz
```

## 7. Fire a grounding action

```bash
# [Laptop]
ros2 action send_goal /semantic/ground_and_navigate \
    go2_semantic_msgs/action/GroundAndNavigate \
    "{text_query: 'go near the chair', stand_off_m: 0.9, dry_run: true}" \
    --feedback
```

Expected feedback progression: `PARSING → SCORING → SAMPLING_GOAL → REACHED` (in dry-run mode). Result includes `chosen_object_label`, `grounding_score`, and `final_goal` in the `map` frame.

## 8. Thermal soak + sustained-rate benchmark

```bash
# [Jetson]
# Run for 10 min under load, log tegrastats in parallel.
tegrastats --interval 1000 --logfile /tmp/tegrastats.log &
# Keep the bringup running in another shell for the full 10 min.
# Then stop:
pkill tegrastats
```

Validate:
- detection FPS sustained ≥3 Hz after 5 min thermal soak
- GPU temp < 85 °C
- no thermal-throttling events in `/var/log/syslog`

## 9. Troubleshooting

See `docs/troubleshooting.md`. Jetson-specific gotchas:

- `libnvinfer.so.10` missing → reinstall `nvidia-tensorrt` for JetPack 6.x
- `sentencepiece` import error → rebuild from source (`pip install sentencepiece --no-binary=:all:`)
- NanoSAM import error → engine files not at `~/nanosam/data/`; set `NANOSAM_ENCODER_ENGINE` and `NANOSAM_DECODER_ENGINE` env vars
- MobileCLIP import error on Jetson → use OpenCLIP ViT-B/16 (same embedding dim, broader ecosystem)
- Thermal throttling under sustained load → lower `detection_rate_hz` to 2.0 or switch to `nvpmodel -m 1` (15 W) and accept lower FPS
