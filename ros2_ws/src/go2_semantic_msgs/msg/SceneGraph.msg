# Snapshot of the open-vocabulary 3D semantic scene graph.
# Published by go2_scene_graph after each merge, keyed to the map frame.

std_msgs/Header header             # frame_id = map_frame; stamp = latest observation incorporated

# Tracked objects (graph nodes).
go2_semantic_msgs/SemanticObject[] nodes

# Pairwise spatial relations (graph edges). Edges are recomputed lazily and
# may lag behind nodes by at most one update cycle.
go2_semantic_msgs/SceneGraphEdge[] edges

# Summary stats (for quick dashboards without iterating nodes).
int32 total_objects
int32 total_edges
float32 median_object_confidence
float32 min_object_confidence
float32 max_object_confidence
