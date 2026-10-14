# Bounded RGB-LiDAR fusion

`go2_fusion_cpp` is an additive C++17 package with a standalone CPU reference, an optional CUDA visibility accelerator, and an isolated ROS 2 adapter. Replay and ROS use the same `Engine::process` implementation. The existing detector, projected-depth implementation, deployed launches, and motion interfaces are unchanged. No motor, navigation, or control commands are published.

The CPU backend is the default. Fusion requires local camera, normalized LiDAR, detection, and TF inputs; it has no remote inference or offboard service dependency. Existing semantic detections remain supported. Their previous 3D values are ignored, and geometry is recomputed from original LiDAR returns. Standard `vision_msgs/Detection2DArray` inputs are also supported. TensorRT remains an optional upstream detector choice; no detector backend is replaced.

## Geometry and validity

The core transforms source-frame points into the camera optical frame at the image timestamp, projects with the five-coefficient Brown `plumb_bob` model, and computes a nearest-depth raster. A local visibility minimum rejects farther returns behind foreground structure. Raster neighborhoods are square and configurable; visibility around object edges is conservative. Original 3D returns are retained, so object association does not invert distorted pixels with a pinhole approximation or quantize geometry into millimetres.

Association uses detector boxes and optional binary ROI masks. Each visible return belongs to at most one detection. Overlapping detections with scores within 0.05 discard shared returns; otherwise the higher score owns them. Support counts unique raster cells, never splat pixels or duplicated returns. Separated depth modes are split at the configured gap; a cluster needs at least six cells and at least 55% of the support by default. Otherwise the result is invalid. Conservative support requirements can reject many objects in sparse single sweeps. Detection correctness itself is not established by geometric support.

The position is the centroid of the selected visible surface returns, not a complete object's centre or physical dimensions. Covariance retains observed surface spread plus common LiDAR, calibration, and timestamp-skew uncertainty. The defaults are conservative model parameters, not empirically calibrated confidence bounds. Rotating to the output frame also rotates covariance. Parameters that overflow uncertainty calculations are rejected; nonfinite output never becomes valid.

A bounded tracker associates observed surface centroids by class and distance in one fixed output frame. Matching is one-to-one, with a 0.6 m gate and 0.5 s lifetime by default. Tracks are not extrapolated into valid measurements during missing data. Output-frame changes clear association history. Monotonically increasing measurement timestamps are required. This tracker does not estimate hidden object shape, motion models, or identity across long occlusions.

The ROS adapter requires all of the following before producing valid geometry:

- Independently validated camera intrinsics and extrinsics, with both YAML files explicitly declaring `nominal: false`.
- A nonempty `calibration_id` identifying the evidence bundle and `calibration_validated: true`.
- `cloud_geometry_validated: true`, attesting the upstream LiDAR frame, sensor calibration, and deskew provenance.
- `timing_validated: true`, attesting source clocks and the camera capture offset.
- Matching raw-image dimensions, frame, K, D, and five-coefficient `plumb_bob` model; unsupported binning, cropping, or skew is rejected.
- Source image and CameraInfo at the detection's exact stamp and frame, plus a normalized cloud within 50 ms by default.
- Timestamp-aware TF from cloud time to camera time through the fixed frame, output TF at image time, and TF matching the measured camera extrinsics.

Setting validation parameters is an operator attestation, not a calibration procedure. Nominal YAML cannot pass the file gate. Trust, frame, backend, and capacity parameters require a node restart to change. Measurement failures carry explicit status, `valid=false`, zero track ID, and NaN position/covariance. Array status reports input/processing validity; each object has its own support/ambiguity status, so a valid array does not imply every object has a position. Consumers must check object validity and age.

Single-sweep inputs are the deployment reference. Accumulated replay is a stress workload only: the current binary format lacks per-point acquisition timestamps, so it cannot establish dynamic-object timing uncertainty across a history window.

## Storage and synchronization

The default core caps clouds at 100,000 points, detections at 64, tracks at 128, and images at 1280 by 720. The CPU core preallocates approximately 14.7 MB. Optional explicit-copy CUDA uses approximately 3.6 MB of core storage plus 12.9 MB of accelerator storage. CUDA process memory also includes driver/context allocations and must be measured separately; allocation sizes alone are not process RSS.

The ROS adapter keeps 16 image metadata/cloud/CameraInfo entries and two pending detection batches by default. RGB payloads are retained only when overlays are enabled. Input payloads and mask batches have separate byte caps. Oversized inputs and pending overflow are rejected; waiting for TF or matching messages has a 50 ms deadline and uses nonblocking TF lookups. Label storage is capped. Queue capacities are bounds, not evidence of lossless delivery under overload. Input middleware deserialization, detector messages, TF storage, and output transport have their own memory costs.

## Optional CUDA

Profiling identified visibility filtering as the CPU bottleneck in recorded workloads. CUDA implements sparse depth rasterization, separable local depth minima, and visibility flags. Camera projection, object-point reduction, uncertainty, and tracking stay on CPU because their measured cost did not justify GPU ports.

The accelerator has preallocated buffers, pinned sparse staging, one nonblocking stream, and one completion synchronization per frame. Transfers contain only `(pixel, depth)` pairs and returned flags; dense depth images remain on device. Every allocation, launch, transfer, and completion status is checked. Failed work is drained and flags cleared; the shared engine invalidates the frame and clears tracks. A requested unavailable backend stops startup. CPU operation remains available by restarting with `backend: cpu`; there is no silent reuse of GPU output.

`backend: cuda_mapped` investigates mapped pinned input when supported. Explicit copy performed better for the sampled single-sweep workload on the development system; this is not a Jetson memory recommendation. Target memory coherency and access behaviour must be measured independently. See [NVIDIA's Tegra memory guidance](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management).

CUDA Graphs were not added. The measured kernels and transfer/launch overhead were already small relative to the requested budget. Graph capture, multi-stream pipelines, GPU tracking, custom detector code, and GPU reduction would add complexity without demonstrated need. Revisit only when target profiling identifies a meaningful bottleneck.

## Build and replay

Build without ROS or CUDA:

```sh
cmake -S ros2_ws/src/go2_fusion_cpp -B /data/evaluation/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /data/evaluation/build -j4
ctest --test-dir /data/evaluation/build --output-on-failure
```

For sanitizer validation, use a separate CPU build with `-DGO2_FUSION_SANITIZE=ON`. Build optional CUDA with `-DGO2_FUSION_CUDA=ON` and the device's `-DCMAKE_CUDA_ARCHITECTURES`; the deployment target uses architecture 87. The conservative default is 86 with forward-compatible PTX. CUDA toolkit availability is required only for the optional build.

[Replay extraction and numerical reference checks](../scripts/fusion/README.md) produce `replay.g2f`, audit reports, and diagnostic overlays. The current `G2FUS001` benchmark uses the recording's fixed nominal 1280x720 camera model, fx=fy=537, cx=639.5, cy=359.5, D=0; it must not be used with a different `camera.json`. The shared core and ROS adapter support supplied camera calibration. Binary geometry is always untrusted by default.

```sh
/data/evaluation/build/fusion_benchmark /data/evaluation/single/replay.g2f \
  30 0 /data/evaluation/projected.csv cpu
/data/evaluation/build/fusion_benchmark /data/evaluation/objects/replay.g2f \
  10 1 - cpu
/data/evaluation/cuda_build/fusion_benchmark /data/evaluation/objects/replay.g2f \
  10 1 - cuda
/data/evaluation/cuda_build/cuda_visibility_benchmark \
  /data/evaluation/objects/replay.g2f 10 both
```

Arguments after the replay are repeats, association exercise flag, optional projection CSV (`-` disables it), and backend. Association exercise uses synthetic trust only to measure the reduction/tracking workload on recorded points and boxes. It exports timing/counts, never object positions or calibrated accuracy. Use flag `0` to verify all nominal geometry remains invalid. Five initial frames per replay are excluded from timing samples. Capacity/skew rejects are counted explicitly. Timings include all `Engine::process` work and CUDA staging/transfers/synchronization, excluding file reads, detector inference, ROS message parsing, TF lookup, publication, and startup. A maximum of one million timing samples is retained.

Percentiles include rejected frames, whose fail-closed path is shorter. The accumulated workload can reject older clouds under the 50 ms gate; inspect invalid counts before comparing percentiles. Projection CSV is frame index 5, with u, v, depth in metres, and visibility. All diagnostic overlays must retain their untrusted-geometry annotation.

## ROS 2

Build only the relevant packages in a separate workspace/output location after sourcing the appropriate ROS underlay:

```sh
colcon --log-base /data/evaluation/ros_log build \
  --base-paths ros2_ws/src --packages-select go2_semantic_msgs go2_fusion_cpp \
  --build-base /data/evaluation/ros_build --install-base /data/evaluation/ros_install \
  --cmake-args -DGO2_FUSION_ROS=ON -DCMAKE_BUILD_TYPE=Release
. /data/evaluation/ros_install/setup.sh
ros2 launch go2_fusion_cpp fusion.launch.py config:=/data/calibration/fusion.yaml
```

The launch starts only fusion and defaults to replay `use_sim_time=true`. Future onboard operation requires explicit `use_sim_time:=false`, validated local sources, and the measured calibration configuration. It does not start drivers, a detector, or deployed services. New output is `/fusion/objects`; optional output is `/fusion/projected_overlay`. Existing source topics are configurable. The existing detector configuration is limited to 5 Hz and still depends on its RGB-D producer. A faster fusion core alone does not establish 10-20 Hz object updates or replace that producer. Independent standard 2D detections can use `input_type: vision`.

Synthetic ROS failures and positive geometry are reproducible after sourcing the generated message packages:

```sh
python3 ros2_ws/src/go2_fusion_cpp/test/test_ros_adapter.py \
  --executable /data/evaluation/ros_install/go2_fusion_cpp/lib/go2_fusion_cpp/fusion_node \
  --results /data/evaluation/ros_faults.json --backend cpu --domain-id 190
```

The suite restricts DDS to localhost and uses synthetic calibration fixtures. Repeat with `--backend cuda` or `cuda_mapped` for optional builds. It does not access hardware or change real calibration.

## Latency and readiness

`fusion_compute_ms` measures the shared core. `adapter_compute_ms` includes input decoding and the processing callback, including the core; it is not additive to fusion compute. `synchronization_wait_ms` excludes that recorded compute work. Full sensor age to publication is the age of the older image/cloud source, measured only when independently validated acquisition clocks and offsets are attested and simulated time is disabled. It excludes downstream DDS delivery and consumer time. Otherwise `sensor_latency_valid=false` and `sensor_to_output_ms=-1`.

Packet-arrival timestamps and bag receipt offsets cannot establish exposure-to-output latency. Separate offline detector timing excludes segmentation and embeddings and cannot be added percentile-wise to fusion timing to derive end-to-end P95/P99.

This implementation is ready for offline evaluation, with synthetic and recorded-input numerical evidence. Deployment requires a complete integrity-verified recording, measured camera calibration and capture offset, upstream LiDAR/deskew provenance, metric ground truth, target CPU/CUDA comparison, sustained thermal/power and memory measurements, detector rate validation, and sensor-to-publication measurements under representative load. Passing workstation compute budgets or replay tests does not prove these gates.
