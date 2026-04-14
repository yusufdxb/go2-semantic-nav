# A directed (or symmetric) spatial-relation edge between two SemanticObject nodes.

string source_object_id
string target_object_id

# Relation type — lowercase_snake. Consumers may add custom types but must
# handle these as canonical:
#   "near"          — undirected; source within near_threshold_m of target
#   "left_of"       — directed; from the robot's current viewpoint
#   "right_of"      — directed; same
#   "in_front_of"   — directed; from robot viewpoint (target is in the cone ahead of source)
#   "behind"        — directed; opposite of in_front_of
#   "above"         — directed; world +Z
#   "below"         — directed; world -Z
#   "on"            — directed; top-surface contact (heuristic)
#   "beside"        — alias for "near" with stricter lateral criterion
string relation_type

float32 distance_m                 # Euclidean centroid distance in map frame
float32 relation_score             # geometric goodness in [0, 1] — how cleanly the relation fits
