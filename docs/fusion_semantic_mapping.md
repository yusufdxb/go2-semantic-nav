# Semantic mapping from validated fusion

The optional `fusion_semantic_adapter` joins `/semantic/detections`
(`go2_semantic_msgs/SemanticDetectionArray`) with `/fusion/objects`
(`go2_fusion_cpp/FusionArray`) and publishes `/semantic/fused_detections`.
The scene graph can consume this stream to retain detector labels and image
embeddings while admitting only validated fused positions.

Build the ROS fusion interfaces and mapper after sourcing ROS 2:

```bash
colcon build --base-paths ros2_ws/src --packages-up-to go2_scene_graph \
    --cmake-args -DGO2_FUSION_ROS=ON
source install/setup.bash
```

CUDA is optional; the default build uses CPU fusion. To write map exports, pass
`snapshot_dir:=/data/semantic_map` to the mapper launch.

```bash
ros2 launch go2_scene_graph scene_graph.launch.py fusion_mapping:=true
```

This starts the adapter and mapper. Start the detector, fusion node, sensor
sources and TF separately. Fusion calibration, LiDAR-geometry and acquisition
timing gates must pass before fusion publishes valid positions. The launch
option defaults to `false`; it does not change an active deployment. Existing
mapper YAML settings remain effective. Optional `map_frame`, `detections_topic`,
`use_sim_time` and `snapshot_dir` launch arguments override only when supplied.
The adapter inherits the mapper frame, semantic input and simulation clock.
Snapshots are output artifacts; this interface does not restore graph state
after a process restart.

Matching requires identical image acquisition stamps, semantic header/source
frames and the fusion camera frame. Detection count, ordered labels and scores
must agree. A batch with failed fusion status is rejected. Objects marked
invalid by fusion are omitted; contradictory status/valid flags reject the
pair. Accepted objects require a nonzero track id, supporting points, finite
positions, symmetric positive-semidefinite covariance and finite nonzero
embeddings with matching dimensions. Original detector depth is never a
fallback. A malformed matched pair produces no mapper observations.

The join supports either arrival order. It holds at most 20 batches per input
for two seconds of monotonic receive time. Duplicate arrivals do not refresh
expiry. Matched stamps cannot be replayed or reordered for a cached camera
frame. Capacity is also bounded to 256 detections and 4,096 embedding components
per object. Parameters `cache_capacity`, `cache_expiry_s`, `max_detections` and
`max_embedding_dim` can reduce these limits. The number of remembered camera
frames is bounded by `cache_capacity`.

## Geometry and transport limits

The output header and `source_frame_id` identify the fusion output position
frame. The legacy `centroid_3d_camera` field carries that fused position in the
stated source frame, allowing the existing mapper to transform it at the image
stamp. `centroid_3d_map_valid` is true only when the output frame equals the
configured `map_frame`; otherwise its map centroid is cleared. This adapter
stream is for the mapper, not a camera-only detection consumer. Preserved 2D
boxes and masks still refer to the original camera image. Backprojection latency
is reset to the unmeasured sentinel `-1` because that geometry was discarded.

`FusionObject` has no object extents or camera depth measurement. The bridge
therefore sets `dimensions_xyz=(0,0,0)` and `depth_median_m=0` as explicit unknown
sentinels. It does not relabel camera-axis extents as map-axis dimensions or
reuse unvalidated detector depth. Zero extents do not establish physical object
size. Dimension-dependent spatial relations and bounding-volume displays need
separately validated extent estimates. Fused covariance is checked before
admission but the legacy semantic message and mapper do not retain covariance.

The mapper still needs TF from the fusion output frame to `map` at the image
stamp unless fusion already outputs map coordinates. This interface does not
establish localization accuracy, cross-frame calibration or full sensor latency.
Offline adapter tests exercise association and rejection contracts; hardware
mapping accuracy requires separate measured validation.
