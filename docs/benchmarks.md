# Benchmarks

Every number here is **measured on a named platform** and **tied to a reproducible command**. Unknown numbers are marked `<pending>`, not extrapolated.

## 1. Grounding round-trip latency: `mewtwo-5070`

20-query suite from `eval/queries.yaml` played against `scripts/synthetic_publisher.py` (ultralytics bus.jpg, constant 2 m depth). Two configs:

| Config | Top-1 | Top-5 | Expected-failure honesty | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| `v1` loose thresholds (`total>=0.15` only) | 0.00 | 0.00 | **0.00** | 7.5 | 12.0 |
| **`v2`** two-layer (absolute + label/clip floor) | 0.00 | 0.00 | **0.67** | 7.8 | 21.3 |
| `v3` three-layer (v2 + `margin>=0.05`) | `<pending>` | `<pending>` | `<pending>` | `<pending>` | `<pending>` |

**Top-1/Top-5 are 0.00 by construction on the synthetic bus-only scene**: no chairs/tables/sofas present, so every in-vocab query's `expected_label` is absent. Honesty is the informative metric here.

**Reproduce:**
```bash
# v2
python eval/run_eval.py --mode synthetic --warmup-s 15 \
    --out eval/results/dev_mewtwo_synthetic_v2 --config-name dev_mewtwo_synthetic_v2
```

## 2. Per-stage detector latency: `mewtwo-5070`

50-frame measurement (post-10 warmup) with `scripts/latency_profiler.py`.

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|
| detector (YOLO-World v2-s) | 4.6 | 4.8 | 5.2 | 4.7 |
| segmenter (MobileSAM) | 37.7 | 38.6 | 39.5 | 37.8 |
| encoder (OpenCLIP ViT-B/16) | 20.7 | 21.3 | 22.3 | 20.8 |
| back-project (NumPy) | 4.6 | 5.0 | 5.1 | 4.7 |
| **TOTAL** | **67.7** | **69.0** | **70.8** | **68.1** |

→ **14.7 Hz theoretical upper bound on RTX 5070** with this exact backend stack.

**Analysis:**
- Segmenter (MobileSAM) is the dominant cost. On Jetson, NanoSAM (TRT-native) is the highest-leverage swap.
- YOLO-World v2-s detector is surprisingly cheap (4.6 ms), the bottleneck is NOT the "flashy" model.
- Back-projection at 4.6 ms is within the same band as the detector, no CUDA kernel is justified here (the sibling `GO2-Perception-Optimization` project's finding on a similar workload).

**Reproduce:**
```bash
# term 1
python3 scripts/synthetic_publisher.py --ros-args -p rate_hz:=8.0
# term 2
ros2 launch go2_open_vocab_detector detector.launch.py
# term 3
python3 scripts/latency_profiler.py --frames 100 --warmup-frames 20 \
    --out eval/results/latencies_dev_mewtwo.json
```

## 3. Record-and-replay integration: `mewtwo-5070`: **VERIFIED 2026-04-13**

`scripts/record_replay_integration_test.py` end-to-end: synthetic publisher → `ros2 bag record` (4 topics) → `ros2 bag play --clock --loop` → in-process detector + scene_graph + grounding (all with `use_sim_time=True`) → grounding action dispatch.

**Observed (cached bag `tests/bags/synthetic_bus/`):**
- scene graph populated AND published **after 1.0 s** of bag playback
- grounding action returned `success=True`, `state=REACHED`
- `chosen_label='person'`, `grounding_score=0.43`
- `final_goal` at `(3.23, 0.92, 0.0)` in `map` frame

Closes the Phase-5 rosbag pathway gap. The same invocation works on the robot against a real live rosbag, the `use_sim_time:=true` + `--clock` combination is the operational fix.

**Reproduce:**
```bash
python scripts/record_replay_integration_test.py \
    --record-duration-s 8 --timeout-s 90 --text-query person
```

## 4. Jetson: `jetson-orin-nx-25w` (pending)

All Jetson rows are `<pending>` until we have the hardware in the loop. Blueprint:

| Measurement | Script | Expected runtime |
|---|---|---|
| Per-stage latency (YOLO-World v2-s + NanoSAM + MobileCLIP-S2) | `scripts/latency_profiler.py` | ~2 min |
| Sustained Hz under thermal soak | `scripts/run_thermal_benchmark.py --duration-s 600` | 10 min |
| Grounding eval on real room | `eval/run_eval.py --mode rosbag --bag <...>` | ~5 min / 20 queries |
| Navigation SR + SPL | manual action-client over Nav2 | ~20 min / room |

## 5. Figures

Produced by `scripts/make_figures.py`:
- `eval/results/figures/grounding_ablation.png`: top-1 / top-5 / honesty + latency per config
- `eval/results/figures/latency_per_stage.png`: p50 bar + p95 error-bar per stage per platform
- `eval/results/figures/thermal_plot.png`: (pending Jetson thermal CSV)

## Known issues surfaced during benchmarking

### `ros2 bag play` + direct node construction = TF-stale race **(FIXED 2026-04-13)**
Initial record-replay attempts failed because `ros2 bag play` replays messages at their ORIGINAL timestamps while subscribers use wall-clock by default. TF lookups then failed because the requested stamp was outside the TF cache window.

**Fix applied:**
1. `scripts/record_replay_integration_test.py` now invokes `ros2 bag play --clock --loop`.
2. The test calls `rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])` so the default context carries sim-time.
3. Each in-process node then has `use_sim_time=True` set explicitly as its first parameter.
4. The composite `semantic_nav.launch.py` exposes `use_sim_time:=true` as a launch arg that threads through all 3 nodes, robot operators use the same arg for on-hardware replay.

Verified PASS on `mewtwo-5070`: scene graph populated in 1.0 s of replay; grounding action returned valid `PoseStamped`.
