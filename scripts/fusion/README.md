# Offline fusion replay tools

These tools preserve the source recording and use the same inputs as the C++ fusion benchmark. They require Python 3.10+, NumPy, PyYAML and `rosbags`. Diagnostic visualizations and visibility comparisons also require OpenCV. Detector sampling uses the project's existing Ultralytics backend and local model weights.

Extract a recording into a separate output directory:

```sh
python scripts/fusion/extract_replay.py \
  --bag /data/recording/recording_0.db3 \
  --output /data/evaluation/single \
  --intrinsics ros2_ws/src/go2_rgb_lidar/config/front_camera_intrinsics_nominal.yaml
```

The extractor audits SQLite integrity, topic counts, header and receipt timestamps, camera information, TF and cloud layouts. It interpolates recorded `odom -> base_link` TF at the RGB timestamp and composes it with recorded static `base_link -> front_camera_optical_frame` TF. It selects the nearest normalized cloud, preserving the signed timestamp difference. Missing bracketing TF, excessive interpolation gaps, unsupported frames and excessive cloud skew reject frames.

`--recover-prefix` permits a diagnostic prefix of a truncated SQLite file. It writes a separate copy with its declared page count limited to physically available pages, then stops at the first unreadable message. This is partial extraction, not database repair or evidence that the complete recording is available. `--recovery-copy /data/evaluation/prefix.db3` chooses the location of that derived copy. The source recording and its directory remain untouched.

Outputs include `audit.json`, `camera.json`, `manifest.json`, `replay.g2f` and sampled projected-point overlays. Replay publication uses an atomic rename after the full output is written. Calibration remains untrusted; these tools never certify nominal or measured geometry. Depth overlays are diagnostic. Receipt offsets do not measure exposure-to-output latency.

`--accumulate-s 0.5` extracts a causal, world-frame cloud history for workload testing. All points use the transform at the image timestamp. The file retains the latest cloud timestamp only, so accumulated inputs cannot establish per-point timing or dynamic-object uncertainty. Use single-sweep inputs for timing-aware association validation until per-point acquisition times are available.

Sample actual detector bounding boxes without changing the detector:

```sh
python scripts/fusion/object_replay.py \
  --replay /data/evaluation/single/replay.g2f \
  --readable-bag /data/evaluation/prefix.db3 \
  --last-readable-message-id 16165 \
  --detector-config ros2_ws/src/go2_open_vocab_detector/config/detector_rgb_lidar.yaml \
  --output /data/evaluation/objects --samples 50 --device cpu
```

Use the message ID reported by `audit.json`, not the illustrative value above. The script applies the configured prompt list, confidence threshold and object limit. `detector.json` records bounding boxes and offline detector wall time after one warmup, including result retrieval. These boxes are predictions, not ground truth or segmentation masks. Detector inference, fusion compute and sensor-to-output latency are separate quantities.

Compare the benchmark's projection CSV with the original Python implementation:

```sh
python scripts/fusion/check_projection.py \
  --replay /data/evaluation/single/replay.g2f \
  --csv /data/evaluation/core_projected.csv --frame 5 \
  --intrinsics ros2_ws/src/go2_rgb_lidar/config/front_camera_intrinsics_nominal.yaml \
  --output /data/evaluation/parity.json
python scripts/fusion/test_extract_replay.py
```

The comparison checks projected point count, pixel coordinates, optical depth and visibility against Python projection and an independent OpenCV local-minimum reference. Coordinate tolerance accounts for C++ float storage and CSV precision. Numerical parity does not establish calibration accuracy.

## Binary format

All fields are little-endian. The camera model and trust metadata live in the accompanying JSON files. Do not infer calibration trust from the binary file.

| Field | Representation |
| --- | --- |
| Magic | 8 bytes, `G2FUS001` |
| Frame count | `uint32` |
| Image and cloud header timestamps, per frame | 2 `int64`, nanoseconds |
| Point count | `uint32` |
| Camera-from-cloud transform | 12 `float64`: row-major rotation 9, translation 3 |
| Points | Point count times 3 `float32`: source-frame XYZ in metres |
| Box count | `uint32` |
| Each box | 5 `float32`: x0, y0, x1, y1, score; then `uint32` label ID |

Boxes use image-edge coordinates with exclusive upper bounds. Full-image extraction uses one diagnostic ROI with score zero and label zero. Detector sampling assigns positive label IDs and saves the label mapping. Nonfinite source points are retained so the runtime must validate them. Extraction caps clouds at 250,000 points; the C++ runner may apply a smaller bound.
