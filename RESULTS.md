# Results

Evaluation numbers for go2-semantic-nav. **Any claim in this file must be measured on the stated platform, not extrapolated.** Ranges in angle brackets are placeholders to be filled after Phase 5 eval runs on real hardware.

## Platforms evaluated

| Id | Description | GPU | CPU | RAM | Power | Notes |
|---|---|---|---|---|---|---|
| `dev-gpu` | Dev workstation | NVIDIA Blackwell consumer GPU | Intel (host) | 32+ GB | ~250 W | Ubuntu 22.04, CUDA 12.8, torch 2.11+cu128 |
| `jetson-orin-nx-25w` | On-robot | NVIDIA Jetson Orin NX 16 GB (Ampere, 1024 CUDA cores) | 8-core ARM Cortex-A78AE | 16 GB unified | 25 W sustained (MAXN) | JetPack 6.x, TRT 10.x, thermal-soaked ≥5 min before measurement |
| `jetson-orin-nx-15w` | On-robot (power-capped) | same | same | same | 15 W sustained | thermal-safe profile |

## Latency (per stage, measured within detector node)

Numbers in the following table are **p50 / p95**. Sample counts and warmup
are stated with each measurement; later fusion benchmarks use a separate protocol.

| Config | Platform | Detector ms | Segmenter ms | Encoder ms | Backproject ms | Total ms |
|---|---|---|---|---|---|---|
| YOLO-Worldv2-s + MobileSAM + OpenCLIP-B/16 | **dev-gpu** | **4.6 / 4.8** | **37.7 / 38.6** | **20.7 / 21.3** | **4.6 / 5.0** | **67.7 / 69.0** |
| YOLO-Worldv2-s + MobileSAM + OpenCLIP-B/16 | jetson-orin-nx-25w | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |
| YOLO-Worldv2-s + NanoSAM + MobileCLIP-S2 | jetson-orin-nx-25w | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |
| YOLOE-11s + NanoSAM + MobileCLIP-S2 | jetson-orin-nx-25w | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |

**dev-gpu measurement (2026-04-13, 50 frames post-10 warmup):** `eval/results/latencies_dev_gpu.json`. Segmenter dominates (MobileSAM encoder runs once per frame + 1 decode per box, 3 boxes here); YOLO-World detector at 4.6 ms is the cheapest stage. Total 68 ms → theoretical 14.7 Hz upper bound on the Blackwell consumer GPU with this exact model choice. All latencies measured inside the detector node's callback via `time.perf_counter_ns()` and published in `SemanticDetectionArray.latency_*_ms`.

**Captured with the three-process pattern (2026-04-13):**
```bash
# term 1 (persistent):
python3 scripts/synthetic_publisher.py --ros-args -p rate_hz:=8.0
# term 2 (persistent):
ros2 launch go2_open_vocab_detector detector.launch.py
# term 3 (runs to completion):
python3 scripts/latency_profiler.py --frames 100 --warmup-frames 20 \
    --timeout-s 120 --out eval/results/latencies_dev_gpu.json
```
The SIGTERM-aware teardown in `synthetic_publisher.py` was added in the same pass to prevent `ExternalShutdownException` noise when processes are killed (e.g., by CI).

### First on-robot measurement (2026-10-02)

Not a table row: 10 detection cycles on one live 1280x720 GO2 front-camera
frame (11 objects) after 2 warm-up cycles, Orin NX in MAXN, no thermal soak,
backends called directly with the node's backend code. torch 2.8.0 (Jetson AI
Lab, JetPack 6 / CUDA 12.6).

| Config | Platform | Detector ms | Segmenter ms | Encoder ms | Total ms |
|---|---|---|---|---|---|
| YOLO-Worldv2-s + MobileSAM + OpenCLIP-B/16 | Orin NX, MAXN | 34.1 / 45.4 (median / max) | 572.7 / 624.6 | 511.1 / 627.3 | 1119 / 1237 |

The segmenter and encoder are 15-25x slower than on dev-gpu, more than the
GPU gap alone; why is not yet measured. Live, the detector node ran near
1 Hz and the scene graph wrote its objects to JSON (see `docs/lab_run.md`).
Object positions use nominal camera calibration and are not validated.

## C++ / CUDA fusion compute on Jetson (2026-10-15)

Three paired rounds compared the previous fusion implementation with the
optimized C++ core and CUDA visibility backend. Each backend replayed the same
50-frame diagnostic input 20 times per round, discarding five warmup frames
per replay: 1,000 processed frames and 900 timed samples per backend per round.
These are repeated samples from the readable prefix of an incomplete recording,
not 1,000 independent captured frames. Backend and variant order were reversed
in the second round. The baseline checkout was `5fced10`, containing the
CPU/CUDA implementation introduced in `a33c366`.

Each latency cell below is the median of three per-run P95 values, followed
by their range in milliseconds. Samples were not pooled. Speedups divide the
baseline P95 by the optimized P95 within each matched round.

| Backend | Baseline P95 ms, median (range) | Optimized P95 ms, median (range) | Matched-round P95 speedup range |
|---|---:|---:|---:|
| C++ CPU | 31.231 (12.390-32.814) | 1.300 (0.980-3.395) | 9.20-33.49x |
| CUDA, explicit copy | 1.265 (0.953-3.213) | 0.532 (0.390-1.228) | 2.38-2.62x |
| CUDA, mapped pinned memory | 2.963 (0.923-3.791) | 2.110 (0.337-2.515) | 1.18-2.74x |

The processes ran at nice priority 10 under changing concurrent stack load.
The baseline CPU maximum reached 630 ms in one round. These observations do
not establish isolated performance or real-time guarantees. Explicit-copy CUDA
had the more consistent P95 behavior in these rounds and is selected by the
optional [Jetson profile](ros2_ws/src/go2_fusion_cpp/config/fusion_jetson.yaml).
The default profile remains CPU.

Working storage fell by **3,686,400 bytes per backend**: CPU core storage from
14,677,120 to 10,990,720 bytes; explicit-copy CUDA accelerator storage from
12,859,200 to 9,172,800 bytes; mapped CUDA accelerator storage from 12,059,200
to 8,372,800 bytes. CUDA modes retain a separate 3,612,800-byte CPU core.
Driver contexts, allocator overhead and total process RSS are additional.

Validation results:

- Jetson Release CTest: 2/2 passed, including 15/15 CPU cases and 590,656
  independent CUDA visibility-flag checks over sparse/dense inputs, borders,
  margin boundaries, duplicates, resizing, extreme radii, empty input and
  failure recovery.
- Recorded replay parity: zero mismatches in 229,510 visibility comparisons
  and 5,980 exact object-result comparisons per CUDA transfer mode. Explicit
  accelerator allocations stayed fixed across 20,000 GPU calls per mode.
- Local AddressSanitizer/UndefinedBehaviorSanitizer: 15/15 CPU cases passed.
  Local CUDA memcheck and synccheck reported zero errors; racecheck reported
  zero hazards.
- Synthetic ROS adapter checks: 21/21 cases for each CPU, CUDA and mapped
  CUDA locally; 21/21 for CUDA on Jetson, including rejection and timing gates.

**Proof boundary:** these results measure fusion compute and tested numerical
parity. The replay uses nominal camera geometry and synthetic trust to exercise
association. It does not validate camera-LiDAR alignment, metric object
positions, acquisition-clock calibration, detector performance, DDS delivery
or full sensor-to-output latency. The 63 one-second telemetry samples peaked
at 66.937 C junction temperature and 19.384 W board power. This short observation
has no thermal-soak qualification or verified power-mode assumption and is
separate from the sustained-throughput protocol below.

Sanitized machine-readable evidence:
[per-run timings](eval/results/fusion_jetson_20261015/paired_comparison.json),
[summary, input hash and source hashes](eval/results/fusion_jetson_20261015/summary.json),
and [parity and test summaries](eval/results/fusion_jetson_20261015/validation.json).
The source recording is not distributed with these results. Implementation and
reproduction commands are in [the fusion engine guide](docs/fusion_engine.md).

## Sustained throughput at Jetson 25 W

Post-thermal-soak (5 min), over 10-min continuous runs.

| Config | Detection Hz | Scene-graph Hz | GPU % | GPU memory MB | GPU temp °C |
|---|---|---|---|---|---|
| Tier-A (YOLO-Worldv2-s + NanoSAM + MobileCLIP-S2) | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |

Thermal plot: `eval/results/latest/thermal_plot.png` (produced by `scripts/benchmark.py`).

## Grounding evaluation

Over 20 queries × 1 trial per config in `eval/queries.yaml` (dev-synthetic = bus.jpg only, hence no chair/table/sofa in scene, most in-vocab queries correctly get `success=False`).

| Config | Top-1 | Top-5 | Expected-failure honesty | Grounding p50 ms | Grounding p95 ms | Notes |
|---|---|---|---|---|---|---|
| dev-A-v1 (dev-gpu, loose thresholds) | 0.00 | 0.00 | 0.00 | 7.5 | 12.0 | baseline; system accepts every weak person-match |
| **dev-A-v2** (dev-gpu, tightened thresholds) | 0.00 | 0.00 | **0.67** | **7.8** | **21.3** | measured 2026-04-13 |
| jetson-A (Tier A) | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` | needs hardware |
| jetson-B (Tier B burst) | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` | needs hardware |

**dev-A-v2 raw output:** `eval/results/dev_gpu_synthetic_v2/summary.json`.

**Note on dev top-1:** 0.00 is the correct result for the synthetic scene, bus.jpg contains only people, but the 20-query suite targets chairs, windows, tables, sofas. A 0 top-1 score for "chair" when no chair is physically in the scene is honest behavior, not a regression; the honesty metric is what's informative here. A real indoor rosbag is needed to measure meaningful top-1 / top-5.

## End-to-end navigation

### Closed-loop simulation (2026-09-18)

Not hardware evidence. `eval/sim_semantic_trials.py` drives text queries
through the full deployment graph (`deploy.launch.py`): grounding ->
NavigateToPose -> Nav2 (LiDAR SLAM localization) -> safety arbiter -> hardware
bridge with the real Unitree Sport API adapter -> `go2_sim`, a kinematic GO2
with the robot's DDS surface, measured robot-clock skew and a simulated L1
LiDAR. Perception is a stand-in (`scripts/sim_scene_graph.py`: labelled world
objects with embeddings at real image-to-text similarity levels), so these
rows measure the grounding-to-actuation chain, not detection. Two-room world
with a 1.0 m doorway; the refrigerator is in the room that starts unmapped.

| Planner | Trials | Present objects reached (correct label, 0.5-1.5 m from surface) | Absent objects refused, no motion | Collisions | Stop distance from surface |
|---|---|---|---|---|---|
| Nav2 (`planner:=nav2`) | 3 | 15/15 | 9/9 | 0 | 0.75-1.28 m |
| Straight-line adapter (`planner:=staged_nav`), open room | 2 | 4/4 | 2/2 | 0 | 0.79-0.96 m |

The straight-line option has no planner and no obstacle avoidance, so it
was run in an obstacle-free room; in the apartment it stops at the first
obstacle on the straight line and holds.

Base-stack goal trials (`ros2 run go2_sim closed_loop_trials`, same graph,
fixed goals through both rooms): 60/60 goals over 12 fresh trials, 0
collisions, arrival error median 0.19 m, p90 0.28 m.

### On the robot

Measured on GO2 EDU with `seeing-eye-dog` Nav2 downstream; ≥3 trials per query.

| Config | Navigation SR | SPL | Mean final-goal error m |
|---|---|---|---|
| Tier-A on robot | `<…>` | `<…>` | `<…>` |

## Ablation

Full 6-configuration ablation table lands in `eval/results/ablation.md`.

## Honest negative results

- **Raw-score thresholding is insufficient on small scenes (documented 2026-04-13).** The v1 grounding used a single `total_score >= 0.15` gate and accepted 16 out of 20 queries against a person-only synthetic scene, even for "approach the guitar" (chose person, total=0.179) and "the dog" (total=0.218). The underlying reason is that OpenCLIP ViT-B/16 gives raw image-text cosines in [0.1, 0.3] for clearly-unrelated pairs (CLIP text-image scores are not calibrated probabilities). The v2 two-layer gate (`absolute_floor=0.15 AND (label>=0.40 OR clip>=0.22)`) lifted expected-failure honesty from 0.00 → 0.67. One out-of-vocab query ("the dog") still slipped through because `CLIP("dog", person_crop) ≈ 0.23 > clip_floor`. A margin-based gate (top-1 − top-2) would catch it but reduces recall on scenes where multiple nearly-indistinguishable objects legitimately satisfy a query. The decision is deferred to Phase 5 post-rosbag eval.
- **Ultralytics YOLO-World has an on-demand CLIP auto-install path** that is not declared in its package metadata. First-run inference triggers `uv pip install git+https://github.com/ultralytics/CLIP.git` which can hang for minutes and does not respect offline environments. Workaround documented in `docs/troubleshooting.md`.
- **Raw `pip install --force-reinstall`** without `--extra-index-url https://download.pytorch.org/whl/cu128` silently replaces the Blackwell-compatible `torch==2.11.0+cu128` wheel with a default CUDA-13 one that fails to run on the Blackwell consumer GPU driver (12.8). Documented in `CONTRIBUTING.md`.

## Reproducibility

All numbers above are reproducible via:

```bash
# dev-A-v2 run (dev-gpu synthetic)
cd go2-semantic-nav
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
python eval/run_eval.py --mode synthetic --warmup-s 15 \
    --out eval/results/dev_gpu_synthetic_v2 \
    --config-name dev_gpu_synthetic_v2

# Jetson run (needs hardware)
# [Jetson] after launching semantic_nav bringup against real camera:
python3 eval/run_eval.py --mode rosbag --bag /data/indoor_a.bag \
    --out eval/results/jetson_A --config-name jetson_A
```

Pinned versions: see `requirements.txt` and `docker/{dev,jetson}.Dockerfile`. Hardware spec: this file. Power mode and thermal state: this file.


## Semantic mapper integration (2026-10-15)

The existing object-centric mapper now has an optional connection from validated
C++/CUDA fusion output. It joins original semantic detections with fusion
results at the exact image stamp and camera frame, preserves visual embeddings,
and admits only valid fused positions. It never falls back to the detector's
original 3D estimate. See [launch and interface contracts](docs/fusion_semantic_mapping.md).

Correctness changes reject malformed geometry before mutating tracked objects
and expire stale objects during graph publication, queries and snapshot export,
even when detections stop. Directed spatial edges now mean
“source relation target”; previously several directions were reversed. Grounding
handles symmetric near/beside edges in either stored orientation. The on
relation requires known extents and approximate surface contact; unknown zero
extents cannot establish contact. This corrects relation behavior without
changing the message schema.

The local Python suite passed 241 tests, including 42 fusion adapter and launch
checks. Package builds completed locally and on Jetson. Synthetic localhost
ROS tests passed locally
and on Jetson: the adapter accepted either input order, replaced deliberately
incorrect detector centroids with fused positions, transformed positions into
map, merged three observations into one object, published the graph, served a
query, wrote JSON snapshots, and expired the object after input stopped. The
archived JSON retained the expired object with live=false. These tests inject
semantic and fusion messages; they do not run a live detector or establish
physical mapping accuracy.

The fused message supplies no object extents or camera depth, so these become
explicit unknown zeros. Covariance is checked on admission but is not retained
by the legacy semantic mapper. JSON snapshots are exports, not restart
checkpoints. The mapper remains a Python reference implementation; the measured
C++/CUDA optimization applies to fusion compute. Accurate live mapping still
requires complete camera-LiDAR alignment, trustworthy acquisition timing,
map-frame localization and held-out object-position measurements.
