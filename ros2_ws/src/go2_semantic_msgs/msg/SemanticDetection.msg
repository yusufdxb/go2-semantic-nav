# One open-vocabulary detection in a single RGB-D frame.
# Published by go2_open_vocab_detector as an element of SemanticDetectionArray.

# --- Identification ---
string label                       # open-vocab prompt that matched (e.g., "chair")
float32 score                      # detector confidence in [0, 1]

# --- 2D geometry (in the source image frame) ---
vision_msgs/BoundingBox2D bbox_2d
sensor_msgs/RegionOfInterest mask_roi   # tight ROI around the mask within the image
uint8[] mask_rle                   # run-length encoded mask over mask_roi (row-major, 0/1)

# --- Language-vision embedding ---
# Unit-normalized CLIP / MobileCLIP image embedding of the object's masked region.
# Dimensionality is backend-dependent; consumers should check `embedding_dim`.
float32[] embedding
int32 embedding_dim                # redundant with len(embedding), but useful for validation

# --- 3D geometry in the camera optical frame ---
geometry_msgs/Point centroid_3d_camera   # meters, in camera_color_optical_frame
float32 depth_median_m                   # robust depth estimate for the object
geometry_msgs/Vector3 dimensions_xyz     # approximate bounding extents in camera frame (m)

# --- Optional 3D geometry in the map frame (set by detector if TF is available) ---
# If the detector node has access to TF, it may pre-transform the centroid here.
# Otherwise the downstream scene-graph node performs the transform.
geometry_msgs/Point centroid_3d_map
bool centroid_3d_map_valid

# --- Frame-local tracking hint ---
# Optional per-frame tracking id; stable within a single SemanticDetectionArray
# but not guaranteed across frames. The scene-graph node does cross-frame association.
string tracklet_id
