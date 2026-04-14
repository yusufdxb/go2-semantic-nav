# Resume bullets — go2-semantic-nav

Tailor length and numbers to the target role. Fill in measured values from the
Jetson ablation once Phase 5 is executed; placeholders are tagged `<…>`.

## Lead bullet (one-line)

> Shipped **open-vocabulary 3D semantic scene-graph mapping and language-grounded navigation** for a Unitree GO2 quadruped on NVIDIA Jetson Orin NX 16 GB: YOLO-World + MobileSAM + OpenCLIP feed a live `map`-frame object graph; an action-server grounds free-form queries ("go next to the red chair") into reachable Nav2 goals. **<SR %>** end-to-end task success on **<N>** queries across **<M>** rooms at **<X>** Hz sustained under a 25 W thermal cap.

## Three-bullet version

- **Designed + shipped a modular ROS 2 overlay (5 packages)** that adds open-vocabulary 3D semantic scene-graph mapping to the Unitree GO2 without editing the existing autonomy stack — subscribes to RealSense RGB-D, publishes `/goal_pose` to Nav2, and exposes a `GroundAndNavigate` action server.
- **Engineered for edge deployment on Jetson Orin NX 16 GB:** pluggable backend layer lets the detector (YOLO-World v2-s / YOLOE-11s), segmenter (MobileSAM / NanoSAM), and CLIP encoder (OpenCLIP ViT-B/16 / MobileCLIP-S2) swap by config; sustained **<X>** FPS detection at 25 W after thermal soak, **<P50>** ms grounding p50.
- **Built an honest evaluation harness** covering grounding top-1/top-5, navigation success, SPL, latency, and expected-failure honesty on a 20-query suite spanning direct nouns, attribute disambiguation (`"the red chair"`), and binary spatial relations (`"the table next to the couch"`). Results, ablations, and failure modes are committed alongside the code.

## Five-bullet version (deep)

- **End-to-end system:** five ROS 2 packages (`go2_semantic_msgs`, `go2_open_vocab_detector`, `go2_scene_graph`, `go2_language_grounding`, `go2_semantic_bringup`) totalling ~<LOC> lines of Python and ROS IDL, deployed as a non-invasive overlay on an existing Unitree GO2 autonomy stack.
- **Online 3D scene graph:** YOLO-World open-vocab boxes + MobileSAM masks + depth back-projection feed an object-centric graph in the `map` frame with embedding- and proximity-based cross-frame association, exponential-moving-average pose smoothing, and lazy spatial-relation edge computation (near / left_of / right_of / in_front_of / behind / above / below / on).
- **Language grounding:** rule-based query parser (handles attribute disambiguation and binary spatial relations) feeds OpenCLIP text embeddings, cosine-ranks candidates against per-object CLIP embeddings, resolves spatial relations geometrically, and samples reachable stand-off poses filtered by the Nav2 costmap.
- **Jetson deployment path:** ONNX + TensorRT FP16 export scripts for YOLO-World (vocab-baked), OpenCLIP image encoder, and MobileSAM (plus a NanoSAM backend for TRT-native Jetson inference); thermal-soaked sustained-rate benchmark and JetPack 6.x deployment cookbook.
- **Research-grade evaluation:** 20-query suite across easy / medium / hard / expected-failure categories; metrics harness reports grounding top-1/top-5, navigation SR + SPL, p50/p95 latency, and expected-failure honesty (correctly refusing queries whose targets are outside the prompt set). Architecture, decisions (including the deliberate choice to drop live 3D Gaussian Splatting in favor of deployable scene graphs), and honest failure modes are documented.

## One-sentence versions (for LinkedIn headline, email subject lines)

- *"Language-grounded quadruped navigation on edge hardware — 3D semantic scene graphs + CLIP + ROS 2, shipped on a Unitree GO2."*
- *"Ship open-vocab robot nav to a Jetson without losing the real-time story."*
- *"Modular ROS 2 overlay that turns text into Nav2 goals for a real quadruped."*

## Interview talking points

- **Why not 3D Gaussian Splatting?** Live GS optimization with densification does not hit real-time on Orin NX 25 W. GS is a rendering representation — the semantic payload still comes from a separate VLM. Object-centric scene graphs (ConceptGraphs, OK-Robot, VLMaps family) match or beat GS on language grounding while being 10–100× cheaper to maintain online.
- **Why overlay, not fork?** The sibling seeing-eye-dog stack is actively being worked on for gait tuning; forking doubles maintenance and splits reviewers. The overlay subscribes to the same camera topics and publishes `/goal_pose` exactly like the existing `go2_intent_grounding` node — zero integration cost, and the overlay repo stands alone as a paper artifact.
- **Hardest-to-debug moment?** TF timing — the scene-graph node was discarding detections because its TF lookup used `stamp=msg.header.stamp` with a 0.1 s timeout, and on a cold boot TF static-transforms hadn't propagated yet. Fixed by retrying with `ros::Time(0)` as a last resort and logging dropped frames at WARN once per 5 s.
- **What would you build next?** A patch-level CLIP feature channel (phase 2.5) — the sibling `openvocab-tsdf` project already does voxel-level CLIP; wiring that in as a Tier-C offboard companion gives queries like "the hallway" or "the empty corner" that object-centric graphs can't answer.
