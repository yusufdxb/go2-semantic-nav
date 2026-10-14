# Header stamp is source image time; frame is the calibrated output frame.
std_msgs/Header header
builtin_interfaces/Time cloud_stamp
string camera_frame
string cloud_frame
string calibration_id
uint8 status
string status_detail
FusionObject[] objects
# -1 when unmeasured because the core was not invoked.
float64 fusion_compute_ms
# Detector-message decoding plus matched-input parsing, TF validation and core.
# Sensor subscription bookkeeping and publication transport are excluded.
float64 adapter_compute_ms
float64 synchronization_wait_ms
# -1 when acquisition clock is not independently validated.
# Sensor age is measured to publication; downstream transport is excluded.
# It is the maximum of image/cloud age and is unavailable under simulated time.
float64 sensor_to_output_ms
bool sensor_latency_valid
float64 image_cloud_skew_ms
uint64 dropped_pending_batches
