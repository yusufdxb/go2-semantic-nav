# Architecture

## System goal

Given a natural-language target description, the robot:

1. Builds and maintains an **object-centric 3D semantic scene graph** in the `map` frame from RGB-D.
2. **Grounds** the language query into a target object and spatial relation.
3. **Plans** a reachable `geometry_msgs/PoseStamped` goal and publishes it on `/goal_pose`.
4. **Executes** using the existing `GO2-seeing-eye-dog` Nav2 + gait-controller stack.

The overlay touches only the perception-and-grounding part of the loop; navigation, safety, and locomotion remain under the sibling stack's ownership.

## Why not 3D Gaussian Splatting

Pure live 3DGS is the wrong core representation on Jetson Orin NX 16 GB at the 25 W thermal cap:

1. **Latency.** Online GS with densification takes multi-second updates under thermal throttling. "Near real-time" becomes dishonest.
2. **Decoupled payload.** GS is *rendering*. The semantic payload that grounds language still comes from a separate VLM (CLIP/OpenCLIP/SigLIP). GS adds cost, not grounding power.
3. **Wrong evaluation surface.** You end up benchmarking PSNR/SSIM, not navigation success.
4. **Research family.** ConceptGraphs (ICRA'24), OK-Robot (RSS'24), VLMaps (ICRA'23), HOV-SG (ICCV'23), CLIO — every recent paper that runs on a real robot uses an object-centric semantic graph. That is the active surface.

The external story — *"real-time semantic 3D mapping + language-guided quadruped navigation on edge hardware"* — is preserved. The primitive is swapped for one that ships.

## Package layout

```
ros2_ws/src/
├── go2_semantic_msgs/       # CMake + IDL (msg/srv/action)
├── go2_open_vocab_detector/ # ament_python; detector lifecycle node
├── go2_scene_graph/         # ament_python; scene-graph lifecycle node
├── go2_language_grounding/  # ament_python; grounding + action server
└── go2_semantic_bringup/    # ament_python; launch files, RViz presets
```

## Data flow

### Perception frame (one RGB-D frame in)

```
image_raw, depth_image_rect_raw, camera_info
    │
    ├─▶ YOLO-World v2 (or YOLOE)  →  N open-vocab boxes with label + score
    │
    ├─▶ MobileSAM (NanoSAM on Jetson) → N masks (RLE)
    │
    ├─▶ OpenCLIP image encoder (MobileCLIP on Jetson) → N × 512-d embeddings
    │
    └─▶ depth back-projection per mask → N × (x,y,z) in camera_color_optical_frame
                                       + per-object depth median
    │
    ▼
SemanticDetectionArray published at detection_rate_hz (default 5 Hz)
```

### Scene-graph update (one detection array in)

```
for each SemanticDetection d in frame F:
    p_cam = d.centroid_3d_camera
    p_map = TF(camera_color_optical_frame → map, stamp=F.stamp).transform(p_cam)
    cand  = [o in graph | dist(o.pose, p_map) < assoc_max_dist_m
                        ∧ cos(o.embedding, d.embedding) ≥ assoc_embed_threshold]
    if cand is empty:
        create new SemanticObject with fresh UUID
    else:
        merge d into argmax_{o ∈ cand} cos(o.embedding, d.embedding)

prune objects not seen in last object_ttl_s seconds
recompute spatial edges (pairwise O(k²) for small graphs; lazy for publish)
publish SceneGraph + MarkerArray
```

### Grounding (one GroundAndNavigate goal in)

```
parse(text) → (target_noun, attribute, relation, reference_noun, stand_off_m)

candidates = top_k_by_score(graph.nodes, target_noun, attribute)
    score(o) = α · cos(text_embed("{attribute} {target_noun}"), o.embedding)
             + β · label_match_indicator(target_noun, o.label)

if relation is not None:
    resolve reference_object (same procedure, filtered set)
    candidates = [c for c in candidates if satisfies_relation(c, reference_object, relation)]

chosen = argmax(candidates)
goal_pose = sample_stand_off(chosen, stand_off_m, costmap=/global_costmap/costmap)
publish /goal_pose (if dry_run=false)
return action result { success, final_goal, chosen_object_id, grounding_score }
```

## Interface contracts

### Subscriptions (mirrors `go2_perception` namespace)
| Topic | Type | QoS |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | BEST_EFFORT, depth 1 |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | BEST_EFFORT, depth 1 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RELIABLE, depth 10 |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | default |

### Publications
| Topic | Type | QoS | Rate / trigger |
|---|---|---|---|
| `/semantic/detections` | `go2_semantic_msgs/SemanticDetectionArray` | RELIABLE, depth 5 | detector timer |
| `/semantic/scene_graph` | `go2_semantic_msgs/SceneGraph` | RELIABLE, depth 5 | after each update |
| `/semantic/object_markers` | `visualization_msgs/MarkerArray` | RELIABLE, depth 5 | with scene_graph |
| `/semantic/grounding_viz` | `visualization_msgs/MarkerArray` | RELIABLE, depth 5 | per grounding request |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RELIABLE, depth 10 | per accepted grounding |

### Services
- `/semantic/query_objects` — `go2_semantic_msgs/srv/QueryObjects`

### Actions
- `/semantic/ground_and_navigate` — `go2_semantic_msgs/action/GroundAndNavigate`

### Required TF chain
`map → odom → base_link → camera_color_optical_frame`

If `/odom` is broken (known blocker on sibling seeing-eye-dog stack), launch with `use_slam_toolbox:=true` to publish `map → odom` via SLAM.

## Model backends

Backends are chosen via the `detector.yaml` config. Current support:

### Detector
- `yolo_world_v2_s` (ultralytics, ~13 M params; AGPL-3.0)
- `yolo_world_v2_m` (ultralytics)
- `yoloe_11s` (ultralytics; AGPL-3.0) — strong Jetson candidate
- `owlv2_base` (HF `google/owlv2-base-patch16-ensemble`; Apache-2.0) — burst mode
- `grounding_dino_tiny` (HF `IDEA-Research/grounding-dino-tiny`; Apache-2.0) — offboard only

### Segmenter
- `mobile_sam` (Apache-2.0) — default on dev
- `nano_sam` (Apache-2.0) — default on Jetson
- `efficient_sam_ti` (Apache-2.0) — fallback
- `sam2_tiny` (Apache-2.0) — burst mode

### Encoder
- `openclip_vit_b16` (MIT) — default on dev
- `mobileclip_s2` (apple-amlr) — default on Jetson (research use)
- `siglip_base` (Apache-2.0) — higher zero-shot accuracy
- `clip_vit_h14` (MIT) — offboard only

See `docs/deployment.md` for per-backend latency tiers and ONNX/TRT notes.

## Configuration surface

`config/detector.yaml`
```yaml
detector:
  backend: yolo_world_v2_s         # yolo_world_v2_{s,m,l,x} | yoloe_11s | owlv2_base
  segmenter: mobile_sam            # mobile_sam | nano_sam | efficient_sam_ti | sam2_tiny
  encoder: openclip_vit_b16        # openclip_vit_b16 | mobileclip_s2 | siglip_base
  device: cuda:0
  detection_rate_hz: 5.0
  conf_threshold: 0.25
  max_objects_per_frame: 30
  prompt_classes:                  # open-vocab prompts
    - chair
    - sofa
    - table
    - window
    - door
    - person
    - refrigerator
    - television
    - bed
    - lamp
  image_topic: /camera/color/image_raw
  depth_topic: /camera/depth/image_rect_raw
  camera_info_topic: /camera/color/camera_info
```

`config/scene_graph.yaml`
```yaml
scene_graph:
  map_frame: map
  camera_frame: camera_color_optical_frame
  assoc_max_dist_m: 0.5
  assoc_embed_threshold: 0.80
  object_ttl_s: 45.0
  min_observations_to_publish: 3
  merge_on_label_match_bonus: 0.05
  spatial_relations:
    near_threshold_m: 1.2
    lateral_cone_deg: 35
```

`config/grounding.yaml`
```yaml
grounding:
  score_weights: {clip: 0.7, label: 0.2, spatial: 0.1}
  stand_off_m: 0.9
  goal_ring_samples: 12
  costmap_topic: /global_costmap/costmap
  use_parser: rules                 # rules | llm
  llm_model: Qwen/Qwen2-0.5B-Instruct
  llm_device: cuda:0
```

## Non-goals (explicit)

- Live 3D Gaussian Splatting or NeRF.
- Custom SLAM (rely on sibling stack's odometry or `slam_toolbox` fallback).
- Gait control or joint-level command.
- Audio input path (voice → grounding query happens outside this repo — see `go2_voice_commander` in sibling stack).
- Multi-robot coordination.
- End-to-end learned navigation policy.

## Failure modes and mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Open-vocab detector misses target class | `/semantic/query_objects` returns empty | Expand `prompt_classes`, or fall back to OWLv2 burst mode on stop |
| CLIP embedding confuses similar objects | Grounding returns wrong chair | Add attribute disambiguation: `score += cos(text("{color} {noun}"), embed)` |
| TF lookup fails | No map-frame objects in graph | Log rate-limited; retry with `slam_toolbox` fallback |
| Goal sample hits obstacle | No reachable pose in ring | Widen ring radius, or abort action with `success=false` |
| Detector too slow on Jetson | FPS < 2 in tegrastats | Swap to smaller backbone (S→N), or reduce frame rate, or pre-compute vocab text embeddings |
| Object tracker drifts | Same object split into multiple graph nodes | Lower `assoc_embed_threshold` or increase `assoc_max_dist_m`; run periodic merge pass |

See `docs/troubleshooting.md` for diagnostic commands.
