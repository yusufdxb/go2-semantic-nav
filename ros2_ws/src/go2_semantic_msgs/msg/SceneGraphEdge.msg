# A spatial-relation edge between two SemanticObject nodes.
# Directed edges read as "source relation target".

string source_object_id
string target_object_id

# Relation type: lowercase_snake. Consumers may add custom types but must
# handle these as canonical:
#   "near"          - symmetric; source within near_threshold_m of target
#   "left_of"       - source left of target in the selected heading frame
#   "right_of"      - source right of target in the selected heading frame
#   "in_front_of"   - source ahead of target in the selected heading frame
#   "behind"        - source behind target in the selected heading frame
#   "above"         - source higher than target along world +Z
#   "below"         - source lower than target along world +Z
#   "on"            - source on target; known extents and approximate surface contact
#   "beside"        - alias for "near"
# Heading defaults to map +X unless an explicit viewpoint heading is supplied.
string relation_type

float32 distance_m                 # Euclidean centroid distance in map frame
float32 relation_score             # geometric goodness in [0, 1]
