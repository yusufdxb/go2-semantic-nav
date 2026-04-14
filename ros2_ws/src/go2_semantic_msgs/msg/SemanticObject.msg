# A tracked object node in the semantic scene graph, expressed in the map frame.
# Maintained by go2_scene_graph and published in SceneGraph.

# --- Identity ---
string object_id                   # stable UUID assigned on first observation
string label                       # most-recent / dominant open-vocab label
float32 confidence                 # running-average detector confidence in [0, 1]

# --- Pose in the map frame ---
geometry_msgs/PoseStamped pose     # stamped with last-update time; frame_id = map_frame
geometry_msgs/Vector3 dimensions_xyz   # extents in meters

# --- Language embedding ---
# Exponential-moving-average of the object's CLIP embedding across observations.
float32[] embedding
int32 embedding_dim

# --- Observation bookkeeping ---
int32 observation_count            # number of frames this object has been associated with
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
float32 last_depth_median_m        # last observed depth (helpful for stale-ness tests)

# --- Alternative labels seen during the object's lifetime ---
# Useful when the open-vocab detector swings between e.g., "chair" and "armchair".
string[] label_history
float32[] label_history_scores
