# A batch of open-vocabulary detections for one RGB-D frame.
# Published by go2_open_vocab_detector at detection_rate_hz.

std_msgs/Header header                     # stamp = image acquisition time; frame_id = camera_color_optical_frame
go2_semantic_msgs/SemanticDetection[] detections

# The source image frame id the detections are expressed in (usually camera_color_optical_frame).
# Redundant with header.frame_id but kept explicit to survive namespace remapping.
string source_frame_id

# The backbone used for the detector (for provenance / eval).
string detector_backend
string segmenter_backend
string encoder_backend

# Latency breakdown in milliseconds, measured within the detector node callback.
# All fields must be filled (-1.0 sentinel if unmeasured).
float32 latency_detector_ms
float32 latency_segmenter_ms
float32 latency_encoder_ms
float32 latency_backproject_ms
float32 latency_total_ms
