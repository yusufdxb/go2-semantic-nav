# GitHub project summary

Copy-paste-ready summary for the repo's About section and for cross-posts on LinkedIn / X.

## One-sentence pitch

> **go2-semantic-nav** — open-vocabulary 3D semantic scene-graph mapping and language-grounded Nav2 goal generation for a Unitree GO2 quadruped on NVIDIA Jetson Orin NX 16 GB, shipped as a non-invasive ROS 2 overlay.

## About / topics

**Topics (for the GitHub About):** `robotics` `ros2` `ros2-humble` `unitree-go2` `jetson` `jetson-orin-nx` `open-vocabulary` `clip` `yolo-world` `mobile-sam` `scene-graph` `language-grounding` `nav2` `tensorrt` `edge-ai`

## Short description

Speak or type a natural-language target — *"go stand next to the red chair"*, *"move near the window"*, *"approach the table beside the couch"* — and the robot:
1. builds a live open-vocabulary 3D object-centric scene graph from RGB-D,
2. grounds the query against it with CLIP text embeddings + a spatial-relation resolver,
3. samples a reachable costmap-aware stand-off pose, and
4. publishes `/goal_pose` into the existing Nav2 stack for execution.

Five ROS 2 packages; overlay design (subscribes to camera topics, does not fork the sibling autonomy stack); pluggable detector / segmenter / encoder backends; ONNX + TensorRT deployment path for Jetson.

## Why this exists

Closed-vocabulary person-following stacks (YOLOv8 on a fixed class list) cannot resolve "the *red* chair" or "approach the window." Live 3D Gaussian Splatting and NeRF-based semantic fields are compelling on paper but do not hit real-time on Jetson Orin NX under a 25 W thermal cap. The deployable middle ground — **object-centric 3D semantic scene graphs with CLIP embeddings** — is what every recent embodied-language-nav paper that actually runs on a robot (ConceptGraphs, OK-Robot, VLMaps, HOV-SG, CLIO) converges on. This repo packages that recipe as a drop-in ROS 2 overlay for a Unitree GO2.

## Featured screenshots (to add)

- RViz with live scene-graph markers + grounding goal arrow over the global costmap.
- Thermal plot: sustained detection FPS vs. time on Jetson Orin NX 25 W.
- Action feedback stream during a grounding dispatch (`PARSING → SCORING → SAMPLING_GOAL → REACHED`).

## Quick links

- [Architecture](docs/architecture.md) — component design, data flow, interface contracts
- [Deployment](docs/deployment.md) — dev + Jetson setup, TensorRT export
- [Jetson cookbook](docs/jetson_cookbook.md) — step-by-step robot bringup
- [Demo](docs/demo.md) — operator procedure for a live walkthrough
- [Experiments](docs/experiments.md) — eval metrics and reproducibility protocol
- [Paper outline](docs/paper_outline.md)
- [Portfolio notes](docs/portfolio_notes.md)
- [AGENTS.md](AGENTS.md) — rules for LLM agents working in this repo

## Related projects in the same ecosystem

- [`GO2-seeing-eye-dog`](https://github.com/yusufdxb/GO2-seeing-eye-dog) — sibling autonomy stack (this repo overlays onto it).
- [`openvocab-tsdf`](../openvocab-tsdf/) — sibling GPU TSDF + voxel-CLIP mapping; serves as an optional Tier-C offboard companion for queries that object-centric graphs can't answer (walls, corridors, empty corners).
- [`GO2-Perception-Optimization`](../GO2-Perception-Optimization/) — sibling profiling + TensorRT + CUDA work; the methodology source for this repo's Jetson benchmarking.
- [`go2-jetson-setup-guide`](../go2-jetson-setup-guide/) — hardware + network bringup for the Jetson side.

## License

MIT for this repo; bundled model weights are under their own licenses (see README footer).
