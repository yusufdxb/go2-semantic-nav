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

## Measured position accuracy

`export_measurements.py` reads `/fusion/objects` from a complete, uncompressed
ROS 2 SQLite bag directory. It registers this repository's message definitions,
so ROS need not be installed. Every listed database must pass SQLite integrity
checks, and the full stream must match metadata message counts. A diagnostic
prefix cannot be used. Output is written atomically outside the bag directory.
Invalid objects and empty arrays are retained.

```sh
python3 scripts/fusion/export_measurements.py \
  --bag /data/evaluation/fusion_bag --output /data/evaluation/measurements.jsonl
python3 scripts/fusion/evaluate_accuracy.py \
  --predictions /data/evaluation/measurements.jsonl \
  --references /data/evaluation/references.jsonl \
  --evidence /data/evaluation/evidence.json --output /data/evaluation/accuracy.json
```

References must be surveyed independently of the fusion estimate and held out
from calibration fitting. Annotate target identities and matching detection
indices using the image, without inspecting estimated XYZ. Schedule reference
frames before evaluating outputs. Include missed detections with
`detection_index: null`; frames without fusion output also remain in the
denominator. Do not select only frames that produced valid positions. There
is no matching by nearest estimated position, timestamp interpolation or frame
conversion in the evaluator.

Each reference JSONL row has the following fields. The numbers below illustrate
the schema only and are not measurements:

```json
{"stamp_ns":1000000000,"target_id":"chair_a","label":"chair","detection_index":0,"frame_id":"odom","position_m":[2.0,0.0,0.5],"position_definition":"visible_surface_centroid","measurement_id":"survey-a","measurement_uncertainty_m":0.01}
```

`position_m` must refer to the same supported visible surface centroid that
fusion estimates. A chair centre, taped nearest point, or floor mark has a
different meaning. Independently survey a flat, clearly isolated surface patch
with known extent and restrict the evaluated observation to that patch, or
obtain an independent 3D surface reference. A taped range to one point alone
cannot validate all three centroid coordinates. The uncertainty field is a
conservative Euclidean distance bound in metres, not a standard deviation.
Its derivation belongs in the survey evidence. If only a range or plane
reference is available, use a separate range or point-to-plane evaluation and
retain the limitation on XYZ accuracy.

The reference and output frames must agree exactly. For static trials, survey
the target relative to a measured robot base origin and transform it into the
chosen fixed frame with documented uncertainty. Moving trials require an
independent trajectory/reference alignment; reusing the estimator's odometry
as motion ground truth does not validate navigation localization. Calibration
changes require new trial evidence and a matching `calibration_id`. Repeated
frames of one pose measure repeatability, not generalization across scenes.

`evidence.json` identifies measured files and the acceptance policy. File paths
are relative to that JSON file's directory. This illustrative policy requires
30 valid samples, at least 95% valid observations, and at most 0.15 m P95 and
maximum error including survey uncertainty. Choose and freeze these criteria
for the application before running the trial.

```json
{
  "data_kind": "measured",
  "calibration_id": "measured-camera-v1",
  "intrinsics_file": "intrinsics.yaml",
  "extrinsics_file": "extrinsics.yaml",
  "timing_evidence_file": "timing.txt",
  "cloud_geometry_evidence_file": "cloud_geometry.txt",
  "ground_truth_evidence_file": "survey.txt",
  "timing_validated": true,
  "cloud_geometry_validated": true,
  "ground_truth_independent": true,
  "ground_truth_used_for_calibration": false,
  "acceptance": {
    "min_valid_samples": 30,
    "min_valid_fraction": 0.95,
    "max_p95_error_m": 0.15,
    "max_error_m": 0.15
  }
}
```

Both calibration YAML files must explicitly declare `nominal: false` and
contain supported finite geometry. Timing and cloud evidence must document
acquisition offsets, clocks, sensor coordinates and deskew provenance, as
required by the fusion adapter. The tool checks documentary prerequisites and
hashes evidence, predictions and references. It cannot verify physical survey
quality or turn an operator attestation into a measured calibration.

The report includes XYZ bias, Euclidean RMSE, median/P95/maximum error,
conservative error bounds including reference uncertainty, valid fraction,
rejection reasons, and per-observation residuals. Unreferenced predictions are
counted but not scored as false positives. Run separate reports for each target
and trial to expose poor coverage. Exit codes are 0 for PASS, 2 for FAIL, 3 for
BLOCKED by missing evidence, and 1 for malformed inputs. PASS applies only to
the recorded observations and stated criteria. Synthetic trials must declare
`data_kind: synthetic` and never receive `metric_accuracy_validated: true`.
It does not prove navigation,
dynamic tracking, live latency or uncertainty coverage.

```sh
python3 scripts/fusion/test_export_measurements.py
python3 scripts/fusion/test_evaluate_accuracy.py
```

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
