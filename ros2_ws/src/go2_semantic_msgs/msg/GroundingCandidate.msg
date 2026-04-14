# A single candidate returned by the grounding scorer, used in QueryObjects responses
# and in GroundAndNavigate feedback.

string object_id                   # id of the matching SemanticObject in the current SceneGraph
string label                       # convenience copy
float32 score                      # total combined grounding score

# Score breakdown (all in [0, 1] before weighting).
float32 score_clip                 # CLIP text-image cosine similarity
float32 score_label                # lexical / wordnet label match
float32 score_spatial              # how well the spatial relation is satisfied

geometry_msgs/PoseStamped object_pose   # same as SemanticObject.pose, copied for convenience
