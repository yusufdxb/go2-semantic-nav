# FAQ

## Why object-centric scene graphs instead of 3D Gaussian Splatting?
Deployability. Live 3DGS with densification is multi-second per update on Jetson Orin NX at 25 W. The semantic payload that grounds language still comes from a separate VLM (CLIP), so GS adds rendering cost without improving grounding. Every recent embodied-language-nav paper that runs on a real robot (ConceptGraphs, OK-Robot, VLMaps, HOV-SG, CLIO) uses object-centric graphs. Full rationale: `docs/architecture.md` §"Why not 3DGS".

## Can this run without a GO2?
Yes. Dev loop runs against `scripts/synthetic_publisher.py` (no hardware, no rosbag) or any RGB-D rosbag with a `map → odom → base_link → camera_color_optical_frame` TF chain. See `scripts/smoke_test_integration.py`.

## How do I add a new detector / segmenter / encoder?
Drop a new module under `ros2_ws/src/go2_open_vocab_detector/go2_open_vocab_detector/backends/`, implement the matching interface from `backends/base.py`, register in `backends/factory.py`. See `CONTRIBUTING.md`.

## Why does grounding return success=True for a query whose target isn't in the scene?
Raw CLIP similarity is noise-dominated when the scene doesn't contain your query class. The default two-layer rejection (absolute floor + label/clip floor) catches most of this, but not all, a query like "the dog" against a person-only scene may slip through because CLIP(dog, person_image) ≈ 0.23. Tighten `clip_floor` or add a margin-based gate (top-1 − top-2) for stricter rejection at the cost of recall.

## Does it speak non-English?
Not out of the box. The rule-based parser is English-only; the LLM parser fallback (`llm_parser.py`) uses Qwen2 which supports English + Chinese. For other languages, swap to SigLIP (multilingual text tower) in `grounding.yaml`.

## How do I handle moving objects (person following, dog)?
Out of scope for this repo, the scene graph is a snapshot, not a tracker. `GO2-seeing-eye-dog` has a working person-follower using closed-vocab YOLO that we explicitly do not duplicate.

## Why 20 queries instead of hundreds?
Honest eval. 20 well-constructed queries with explicit `easy/medium/hard/expected-failure` categorization tell you more than 200 shallow ones, especially when you also measure expected-failure honesty (refusing to guess on out-of-vocabulary targets). Scale up after the baseline is stable.

## What's Tier-A / B / C?
Deploy profiles in `docs/architecture.md` §"Model backends":
- **Tier A (primary, onboard Jetson 25 W):** YOLO-World v2-s + NanoSAM + MobileCLIP-S2 @ ≥3 Hz.
- **Tier B (burst, onboard, stationary scan):** OWLv2-base + SAM 2 tiny + MobileCLIP-S2 @ ~1 Hz.
- **Tier C (offboard Blackwell consumer GPU companion):** Grounding-DINO-B + SAM 2 base+ + OpenCLIP ViT-H/14 via DDS.

## What's the relationship to openvocab-tsdf?
Sibling project. `openvocab-tsdf` is the desktop-class GPU TSDF + voxel-CLIP side; this repo is the robot-side Jetson-deployable object-centric side. Tier-C offboard companion integration is on the roadmap.

## Can I use this commercially?
MIT for our code. But bundled model weights have their own licenses: Ultralytics wrappers (YOLO-World, YOLOE) are AGPL-3.0 (taints derivative firmware); MobileCLIP is `apple-amlr` (research OK, check for products). For commercial deploy, swap to OWLv2 (Apache-2.0) + SAM (Apache-2.0) + OpenCLIP (MIT).
