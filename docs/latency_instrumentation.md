# Latency instrumentation

Per-stage latency telemetry is baked into `SemanticDetectionArray` so every frame
carries its own timing trace. That lets you benchmark the pipeline in situ
without attaching a profiler.

## What's measured

Populated by `go2_open_vocab_detector/detector_node.py` on the publisher side:

| Field | Unit | What it covers |
|---|---|---|
| `latency_detector_ms` | ms | Detector backend's `detect()` call (includes preprocessing + backend inference + postprocessing) |
| `latency_segmenter_ms` | ms | Segmenter backend's `segment()` call (image encode + one decode per box) |
| `latency_encoder_ms` | ms | Encoder backend's `encode_images()` call (batched across all detected boxes) |
| `latency_backproject_ms` | ms | Depth back-projection + RLE encoding for all detections in the frame |
| `latency_total_ms` | ms | Detector-node callback entry → publish (end-to-end detector-side) |

Sentinel `-1.0` in any field means "unmeasured" (legacy frames from older publisher).

## Measurement method

Backends use `time.perf_counter_ns()` around their own work and return it inside
`DetectorOutput.latency_ms` / `SegmenterOutput.latency_ms` / `EncoderOutput.latency_ms`.
The node aggregates those and adds its own back-projection timing around the depth
conversion loop.

Intentionally **not** measured here:

- ROS 2 message serialization and DDS transport (use `header.stamp` → subscriber-side
  timestamps to derive transport latency; `scripts/latency_profiler.py` has an
  option to do this).
- Scene-graph merge + publish latency (log via the scene-graph node's own rclpy
  debug output).
- Nav2 planning time (nav2 has its own telemetry).

## How to capture a dataset

```bash
# Alongside a live or replayed semantic_nav stack:
python scripts/latency_profiler.py --frames 1000 --warmup-frames 100 \
    --out eval/results/latencies_$(date +%Y%m%d_%H%M%S).json
```

Output JSON structure:
```json
{
  "latency_detector_ms":  {"n": 1000, "mean": 64.3, "p50": 58.1, "p95": 102.4, "p99": 174.8, "std": 12.1, "min": 39.2, "max": 240.3},
  "latency_segmenter_ms": {...},
  ...
}
```

## Rules for reporting

- **Always state platform + power mode.** A p95 without "Jetson Orin NX 25 W sustained" is meaningless.
- **Warmup** is mandatory. Discard ≥100 frames before the measurement window.
- **Thermal soak** is mandatory on Jetson. ≥5 min at operating load before measurement.
- **No cherry-picking.** Report the full measurement window p50/p95/p99, never a burst.
- **Include the frame rate you drove the detector at.** A 60-Hz publisher hitting a 200 ms stage will show queuing, not detection latency.
- **Compare against optimized NumPy / PyTorch before claiming CUDA / TRT wins.** A TRT INT8 run that's 3× faster than unoptimized PyTorch FP32 proves nothing; compare to FP16 PyTorch.

## Example honest table row (template)

| config | platform | warmup | frames | detector p50/p95 | total p50/p95 | sustained Hz |
|---|---|---|---|---|---|---|
| YOLO-World v2-s + MobileSAM + OpenCLIP ViT-B/16 | mewtwo-5070 (250 W, CUDA 12.8) | 100 | 1000 | 22 / 41 ms | 58 / 94 ms | 12.3 Hz |
| YOLO-World v2-s + NanoSAM + MobileCLIP-S2 | jetson-orin-nx-25w (post-soak) | 100 | 1000 | `<…>` | `<…>` | `<…>` |

(The first row is to be filled with numbers from `run_eval.py --mode synthetic`; the second row requires hardware.)

## When the numbers look wrong

- Total ≪ sum of stages → backends are overlapping (good) — but check that your
  executor is actually parallelizing them.
- Detector p95 ≫ p50 with big tail → GPU thermal throttling, check `tegrastats`.
- Segmenter dominates → too many detections; cap `max_objects_per_frame`.
- Encoder dominates → batch size too small; check that you're encoding all boxes
  in one forward, not one-at-a-time.
- Backproject ≫ 10 ms → depth image not aligned to color (triggers `cv2.resize`).
  Enable `align_depth.enable:=true` in the RealSense driver.
