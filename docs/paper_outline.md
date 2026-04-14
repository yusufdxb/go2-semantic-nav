# Paper outline — go2-semantic-nav

Venue candidates (ordered by fit): **RSS workshop** (late submission) → **ICRA** → **IROS** → **CoRL workshop** → **RA-L**. A workshop submission is the realistic first-pass target.

## Working title

*Object-Centric 3D Semantic Scene Graphs for Language-Grounded Quadruped Navigation on Edge Hardware*

Alt title (less aggressive): *Deployable Open-Vocabulary Language Grounding for a Unitree GO2 with a Jetson Orin NX*

## Abstract (draft, ~150 words)

> We present an open-vocabulary 3D semantic scene-graph system for language-grounded navigation on a Unitree GO2 quadruped equipped with an NVIDIA Jetson Orin NX 16 GB and an Intel RealSense D435i. The system pairs a YOLO-World v2-s open-vocabulary detector with MobileSAM segmentation and OpenCLIP ViT-B/16 per-object embeddings, lifts each detection into the robot's `map` frame via RGB-D back-projection, and maintains an object-centric graph with embedding- and proximity-based cross-frame association. A lightweight rule-based query parser combined with CLIP text embeddings and a geometric spatial-relation resolver converts natural-language targets such as *"go stand next to the red chair"* into reachable Nav2 goals within **<P95>** ms p95. On a 20-query evaluation across **<M>** indoor rooms, the system achieves **<top1>** top-1 grounding and **<SR>** navigation success while sustaining **<X>** Hz detection under the Jetson's 25 W thermal budget. We report ablation results across detector, segmenter, and encoder variants, document honest failure modes (including out-of-prompt-vocabulary refusals), and release the full stack as a non-invasive ROS 2 overlay.

## 1. Introduction

- Problem: quadruped navigation with natural-language targets in previously unseen indoor environments on edge hardware.
- Existing gap: closed-vocab person-following (our own sibling stack `GO2-seeing-eye-dog`) cannot resolve attribute-qualified or spatially-relational queries; live 3DGS / NeRF-based methods do not hit real-time on Orin NX under thermal load.
- Our thesis: **object-centric 3D semantic scene graphs + CLIP embeddings, assembled as a non-invasive ROS 2 overlay, are the deployable substitute.**
- Contributions (short list):
  1. A modular ROS 2 system with 5 packages that overlays cleanly on existing autonomy stacks.
  2. A pluggable detector / segmenter / encoder architecture with measured latency profiles on Orin NX.
  3. An honest evaluation suite and ablation on 20 queries spanning direct nouns, attribute disambiguation, and binary spatial relations — including expected-failure (out-of-vocabulary) honesty.

## 2. Related work

- **Embodied language-nav via scene graphs:** ConceptGraphs (ICRA'24), OK-Robot (RSS'24), VLMaps (ICRA'23), HOV-SG (ICCV'23), CLIO.
- **Dense neural field representations:** LeRF, Gaussian-Splatting-SLAM, LangSplat — compare compute cost.
- **Open-vocabulary detection:** YOLO-World v1/v2, YOLOE, OWLv2, Grounding-DINO.
- **Efficient segmentation:** SAM, MobileSAM, NanoSAM, EfficientSAM, SAM 2.
- **CLIP-family encoders:** OpenCLIP, SigLIP, MobileCLIP.
- **Quadruped autonomy:** prior Unitree GO2 work including our own sibling stacks.

## 3. System design

### 3.1 Overlay pattern
- The overlay subscribes only to camera topics and publishes `/goal_pose`; no edits to the sibling autonomy stack.
- Five packages, Python-only, with an IDL package that bridges custom messages.

### 3.2 Perception frame
- Detector (YOLO-World v2-s) → segmenter (MobileSAM) → encoder (OpenCLIP ViT-B/16).
- Depth back-projection per mask to camera-frame 3D.
- `SemanticDetectionArray` published at 3–5 Hz with per-stage latency telemetry.

### 3.3 Scene graph
- Cross-frame association: `dist(p, p') < τ_d ∧ cos(e, e') ≥ τ_e`, tie-broken by distance.
- EMA pose/embedding updates, TTL pruning.
- Lazy spatial-relation edge computation at publish time.

### 3.4 Language grounding
- Rule-based parser extracts `(target_noun, attribute, relation, reference_noun, stand_off)`.
- Score = α · CLIP(text, embed) + β · label_match + γ · spatial_fit.
- Attribute disambiguation via relative-prompt delta (`cos("red chair", emb) − cos("chair", emb)`).
- Costmap-aware stand-off ring sampling with facing-toward-target yaw.

## 4. Implementation

- ROS 2 Humble, Python 3.10, rclpy MultiThreadedExecutor per node.
- QoS: camera image topics BEST_EFFORT depth 1; semantic publications RELIABLE depth 5; goal RELIABLE depth 10.
- Pluggable backend factory for detector/segmenter/encoder — 9 concrete backends in-tree.
- ONNX + TensorRT export scripts for Jetson deploy.
- Dev container (x86 CUDA 12.8) and Jetson container (JetPack 6.x l4t-pytorch).

## 5. Evaluation

### 5.1 Experimental setup
- Robot: Unitree GO2 EDU.
- Compute: Jetson Orin NX 16 GB (deploy) + RTX 5070 (dev, comparison only).
- Sensor: Intel RealSense D435i (RGB 640×480 @ 30 Hz, depth rectified).
- Rooms: mewtwo office (15 m²), lab common area (40 m²), held-out apartment living room (25 m²).

### 5.2 Query suite (20 items)
- 3 direct nouns (easy)
- 4 attribute-qualified nouns (medium)
- 3 unary spatial relations (medium)
- 5 binary spatial relations (hard)
- 3 view-dependent deictic (hard)
- 2 out-of-vocabulary (expected failure)

### 5.3 Metrics
- Grounding top-1 / top-5
- Navigation success rate (SR), success weighted by path length (SPL)
- End-to-end task success (human-judged)
- Grounding latency p50 / p95
- Sustained detection rate at Jetson 25 W after thermal soak
- Expected-failure honesty (fraction of out-of-vocab queries correctly refused)

### 5.4 Ablations (5 × 20 = 100 trials minimum)
| Config | Detector | Segmenter | Encoder | Platform |
|---|---|---|---|---|
| A1 | YOLO-World v2-s | MobileSAM | OpenCLIP ViT-B/16 | RTX 5070 |
| A2 | YOLO-World v2-s | NanoSAM | MobileCLIP-S2 | Jetson 25 W |
| B1 | YOLOE-11s | NanoSAM | MobileCLIP-S2 | Jetson 25 W |
| B2 | OWLv2-base | SAM 2 tiny | MobileCLIP-S2 | Jetson 25 W (burst) |
| C1 | Grounding-DINO-B | SAM 2 base-plus | OpenCLIP ViT-H/14 | RTX 5070 (offboard) |

### 5.5 Results
- Per-config ablation table with 95% CIs.
- Thermal plot: sustained FPS vs. time over 10 min on Jetson.
- Latency histogram (p50 / p95 / p99) per stage.
- Negative result: an ablation where the larger detector (OWLv2) gave better recall but worse end-to-end grounding due to embedding-misalignment with MobileCLIP — documented, not hidden.

## 6. Discussion

- When object-centric scene graphs fail: texture-less walls, furniture occlusions, query targets outside the prompt set. Our out-of-vocab honesty metric surfaces this cleanly.
- Sibling offboard companion (`openvocab-tsdf`): voxel TSDF + CLIP feature aggregation handles queries like "the corner" that our graph cannot.
- Why not 3DGS: we ran the numbers and document them.

## 7. Limitations and future work

- No closed-loop re-grounding during navigation (currently "pick-and-commit"). A replanning loop at 1 Hz is tractable.
- English only, single-turn queries. Multi-turn disambiguation ("which chair?" → "the red one") would benefit from an LLM parser.
- No dynamic-obstacle adaptation for moving targets (dog, person).
- License-constrained models (MobileCLIP, YOLO-World via Ultralytics) require eventual replacement for commercial deploy.

## 8. Conclusion

We demonstrate that an object-centric 3D semantic scene graph plus an open-vocabulary perception stack is sufficient — and notably, more deployable than dense neural-field alternatives — to ground natural-language navigation targets on a real quadruped under a 25 W Jetson thermal cap. The full system is released as a non-invasive ROS 2 overlay.

---

## Figures to produce (Phase 5)

1. System block diagram (from `docs/architecture.md`, cleaned up).
2. RViz screenshot with scene-graph spheres + grounding target arrow over the global costmap.
3. Thermal plot: sustained FPS over 10 min on Jetson 25 W.
4. Ablation bar chart across 5 configs × 20 queries (grounding top-1).
5. Latency histogram (p50 / p95 / p99) per stage, Jetson vs RTX.
6. Failure-mode gallery: 3 representative misgroundings with a 1-line analysis each.
