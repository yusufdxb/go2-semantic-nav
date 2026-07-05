# System diagram

## Full data flow (as of 2026-04-13)

```mermaid
flowchart LR
  subgraph hw["GO2 + Jetson Orin NX"]
    RS[RealSense D435i]
    JET[Jetson]
  end

  subgraph edd["GO2-seeing-eye-dog (sibling; we overlay it)"]
    RSDRV[realsense2_camera driver]
    AMCL[AMCL / TF]
    NAV2[Nav2 stack]
    GAIT[go2_gait_controller]
  end

  subgraph ovr["go2-semantic-nav overlay (this repo)"]
    DET[go2_open_vocab_detector<br/>YOLO-World / MobileSAM / OpenCLIP]
    SG[go2_scene_graph<br/>object-centric graph in map frame]
    GR[go2_language_grounding<br/>parser + CLIP + goal sampler]
  end

  subgraph user["Operator"]
    CLI[ros2 action send_goal<br/>or scripts/send_query.py]
  end

  RS --> RSDRV
  RSDRV -- "/camera/color/image_raw<br/>/camera/depth/image_rect_raw<br/>/camera/color/camera_info" --> DET
  RSDRV -- depth cloud --> NAV2

  AMCL -- "/tf map→odom→base_link→camera_color_optical_frame" --> DET
  AMCL -. TF .-> SG
  AMCL -. TF .-> NAV2

  DET -- "/semantic/detections<br/>(SemanticDetectionArray)" --> SG
  SG -- "/semantic/scene_graph<br/>(SceneGraph)" --> GR
  SG -- "/semantic/object_markers<br/>(MarkerArray)" -.-> RVIZ[RViz]

  CLI -- "GroundAndNavigate.action" --> GR
  GR -- "/goal_pose<br/>(PoseStamped map)" --> NAV2
  GR -- "/semantic/grounding_viz" -.-> RVIZ

  NAV2 -- "/cmd_vel" --> GAIT
  GAIT -- "joint_trajectory" --> JET

  classDef over fill:#253,color:#fff,stroke:#9f9
  class DET,SG,GR over
```

## Backend plug points

```mermaid
flowchart TD
  subgraph det["go2_open_vocab_detector"]
    D[DetectorBackend]
    S[SegmenterBackend]
    E[EncoderBackend]
  end
  D --- YW[YOLO-World v2-s/m/l/x]
  D --- YE[YOLOE-11s]
  D --- OW[OWLv2 base/large]
  D --- GD[Grounding-DINO tiny/base]
  S --- MS[MobileSAM]
  S --- NS[NanoSAM]
  S --- ES[EfficientSAM Ti/S]
  S --- S2[SAM 2 tiny/small/base+]
  E --- OC[OpenCLIP ViT-B/16, L/14, H/14]
  E --- MC[MobileCLIP S0/S1/S2/B]
  E --- SL[SigLIP base / SO400M]

  classDef active stroke:#9f9,stroke-width:3px
  class YW,MS,OC active
```
Bold-outlined backends are wired into the default profile (`config/scene_profiles/dev_gpu.yaml`). Jetson Tier A swaps to `NS` + `MC-S2`.

## Goal-generation flow (inside grounding node)

```mermaid
flowchart TD
  Q[text query] --> P[rule-based parser<br/>→ target_noun, attribute, relation, ref_noun]
  P --> TE[CLIP text encode<br/>of full query + plain noun]
  TE --> SC[score every SceneGraph node<br/>= 0.7·CLIP_delta + 0.2·label_match + 0.1·spatial_fit]
  SC --> REL[apply relation filter<br/>if parsed.relation is set]
  REL --> G1{3-layer rejection}
  G1 --> |absolute floor fail| FAIL[success=false<br/>GROUNDING_FAILED]
  G1 --> |weak on label AND clip| FAIL
  G1 --> |margin < margin_min| FAIL
  G1 --> |pass| SMP[sample stand-off ring around chosen object]
  SMP --> CM{costmap gate<br/>if enabled}
  CM --> |no free cell| FAIL2[success=false<br/>COSTMAP_UNREACHABLE]
  CM --> |reachable pose| PUB[publish /goal_pose<br/>unless dry_run]
  PUB --> OK[success=true<br/>state=NAVIGATING or REACHED]
```

## Topic + QoS contract (current live)

| Direction | Topic | Type | Reliability | Depth | Source |
|---|---|---|---|---|---|
| consume | `/camera/color/image_raw` | `sensor_msgs/Image` | BEST_EFFORT | 1 | RealSense driver |
| consume | `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | BEST_EFFORT | 1 | RealSense driver |
| consume | `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RELIABLE | 10 | RealSense driver |
| consume | `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | default | default | AMCL / robot state |
| consume | `/global_costmap/costmap` | `nav_msgs/OccupancyGrid` | RELIABLE | 1 | Nav2 |
| produce | `/semantic/detections` | `go2_semantic_msgs/SemanticDetectionArray` | RELIABLE | 5 | detector |
| produce | `/semantic/scene_graph` | `go2_semantic_msgs/SceneGraph` | RELIABLE | 5 | scene_graph |
| produce | `/semantic/object_markers` | `visualization_msgs/MarkerArray` | RELIABLE | 5 | scene_graph |
| produce | `/semantic/grounding_viz` | `visualization_msgs/MarkerArray` | RELIABLE | 10 | grounding |
| produce | `/goal_pose` | `geometry_msgs/PoseStamped` | RELIABLE | 10 | grounding |
| service | `/semantic/query_objects` | `go2_semantic_msgs/QueryObjects` |, |, | scene_graph |
| action | `/semantic/ground_and_navigate` | `go2_semantic_msgs/GroundAndNavigate` |, |, | grounding |
