# go2-semantic-nav

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Jetson Orin NX](https://img.shields.io/badge/Jetson-Orin_NX_16GB-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

**Open-vocabulary 3D semantic scene-graph mapping and language-grounded navigation for the Unitree GO2 quadruped on NVIDIA Jetson Orin NX 16 GB.**

Speak or type a natural-language target — *"go stand next to the red chair"*, *"move near the window"*, *"approach the table beside the couch"* — and the robot builds an open-vocabulary 3D scene graph online, grounds the query against it, computes a reachable goal pose, and hands off to the existing Nav2 stack for execution.

This repository is a modular **overlay** on top of the [`GO2-seeing-eye-dog`](https://github.com/yusufdxb/GO2-seeing-eye-dog) autonomy stack. It does not replace any existing nodes; it subscribes to the RealSense camera topics, publishes `/goal_pose` in the `map` frame, and leaves Nav2, the gait controller, and the safety monitor untouched.

---

## Why this project

Closed-vocabulary perception (YOLO on fixed classes) cannot resolve "the *red* chair" or "the window". Recent embodied-nav research — ConceptGraphs (ICRA'24), OK-Robot (RSS'24), VLMaps (ICRA'23), HOV-SG (ICCV'23), CLIO — has converged on **object-centric 3D semantic scene graphs with CLIP embeddings** as the deployable substitute for dense neural fields and 3D Gaussian Splatting. This repo packages that pattern as a clean ROS 2 plugin for a real quadruped.

### Why not 3D Gaussian Splatting?
Live 3DGS optimization with densification does not hit near-real-time on Jetson Orin NX 16 GB at the 25 W thermal cap. GS is a *rendering* representation; the semantic payload still comes from a separate VLM. Swapping to an object-centric graph preserves the external story ("real-time 3D semantic mapping + language-guided quadruped nav on edge hardware") while dramatically raising the odds of a finished, demoable system. See `docs/architecture.md` for the full rationale.

---

## System at a glance

```
RealSense D435i (GO2)
 └── /camera/color/image_raw, /camera/depth/image_rect_raw, /camera/color/camera_info
           │
           ▼
 ┌──────────────────────────────────────────────┐
 │ go2_open_vocab_detector  (rclpy lifecycle)   │
 │   YOLO-World v2 / YOLOE — open-vocab boxes   │
 │   MobileSAM / NanoSAM   — mask per box       │
 │   OpenCLIP / MobileCLIP — per-object embed   │
 │   depth back-projection — 3D centroid        │
 │   publishes: /semantic/detections            │
 └──────────────────────────────────────────────┘
           │
           ▼
 ┌──────────────────────────────────────────────┐
 │ go2_scene_graph          (rclpy lifecycle)   │
 │   TF camera_color_optical_frame → map        │
 │   data-association + merge across frames    │
 │   publishes: /semantic/scene_graph           │
 │              /semantic/object_markers (RViz) │
 │   serves:    /semantic/query_objects         │
 └──────────────────────────────────────────────┘
           │
           ▼
 ┌──────────────────────────────────────────────┐
 │ go2_language_grounding   (rclpy lifecycle)   │
 │   query parser + CLIP text encode            │
 │   spatial relation resolver                  │
 │   costmap-aware stand-off goal sampling      │
 │   action: /semantic/ground_and_navigate      │
 │   publishes: /goal_pose  →  Nav2             │
 └──────────────────────────────────────────────┘
           │
           ▼
  [ existing Nav2 + go2_gait_controller from GO2-seeing-eye-dog ]
```

Five ROS 2 packages (four Python, one CMake/IDL):

| Package | Role |
|---|---|
| `go2_semantic_msgs` | Custom messages, service, and action |
| `go2_open_vocab_detector` | Open-vocab detection + segmentation + CLIP embedding, publishes per-frame `SemanticDetectionArray` |
| `go2_scene_graph` | Online 3D object-centric scene graph in `map` frame with spatial relations |
| `go2_language_grounding` | Text query → chosen object → reachable `PoseStamped` on `/goal_pose`; exposes an action server |
| `go2_semantic_bringup` | Launch orchestration, RViz preset, composite configs |

---

## Prerequisites

- **ROS 2 Humble** with `nav2_bringup` (already present on `mewtwo`)
- **Python 3.10** (matches JetPack 6.x)
- **PyTorch 2.x with CUDA 12.x** (tested with `torch==2.11.0+cu128` on RTX 5070)
- **RealSense camera** publishing on `/camera/...` (same namespace as `go2_perception`)
- **TF tree** with `map → odom → base_link → camera_color_optical_frame`
- **For deployment:** a running `GO2-seeing-eye-dog` overlay so Nav2 is up

Install Python ML deps into the colcon workspace venv:

```bash
pip install -r requirements.txt
```

## Build

```bash
cd ~/Projects/personal/go2-semantic-nav/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Run (dev, with a recorded rosbag)

```bash
# Terminal 1 — play the recorded RGB-D bag
ros2 bag play ~/data/go2-indoor-sample/

# Terminal 2 — launch semantic-nav stack (detector + scene graph + grounding)
ros2 launch go2_semantic_bringup semantic_nav.launch.py \
    device:=cuda:0 \
    prompt_classes_file:=$(ros2 pkg prefix go2_open_vocab_detector)/share/go2_open_vocab_detector/config/prompts_indoor.yaml

# Terminal 3 — ground a query
ros2 action send_goal /semantic/ground_and_navigate \
    go2_semantic_msgs/action/GroundAndNavigate \
    "{text_query: 'go near the red chair', stand_off_m: 0.9, dry_run: true}" \
    --feedback
```

`dry_run: true` computes and returns the goal pose without publishing to Nav2. Drop it to trigger navigation.

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Component design, data flow, interface contracts, known constraints |
| [`docs/deployment.md`](docs/deployment.md) | Jetson install, TensorRT export, launch on the robot |
| [`docs/demo.md`](docs/demo.md) | Operator procedure for a live demo run |
| [`docs/experiments.md`](docs/experiments.md) | Eval suite, metrics, how to reproduce reported numbers |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Common failure modes and fixes |
| [`AGENTS.md`](AGENTS.md) | Rules for agents working in this repo |

## Project status

Phase 0 — scaffolding — in progress. See [`docs/experiments.md`](docs/experiments.md) and the Obsidian project notes for the current milestone and task list.

## License

MIT. See [LICENSE](LICENSE). Note that bundled model weights are licensed separately (Apache-2.0 for OpenCLIP and SAM variants; AGPL-3.0 for Ultralytics wrappers; `apple-amlr` for MobileCLIP — research use OK, check before any commercial deployment).
