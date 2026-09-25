# Results

Evaluation numbers for go2-semantic-nav. **Any claim in this file must be measured on the stated platform, not extrapolated.** Ranges in angle brackets are placeholders to be filled after Phase 5 eval runs on real hardware.

## Platforms evaluated

| Id | Description | GPU | CPU | RAM | Power | Notes |
|---|---|---|---|---|---|---|
| `dev-gpu` | Dev workstation | NVIDIA Blackwell consumer GPU | Intel (host) | 32+ GB | ~250 W | Ubuntu 22.04, CUDA 12.8, torch 2.11+cu128 |
| `jetson-orin-nx-25w` | On-robot | NVIDIA Jetson Orin NX 16 GB (Ampere, 1024 CUDA cores) | 8-core ARM Cortex-A78AE | 16 GB unified | 25 W sustained (MAXN) | JetPack 6.x, TRT 10.x, thermal-soaked ≥5 min before measurement |
| `jetson-orin-nx-15w` | On-robot (power-capped) | same | same | same | 15 W sustained | thermal-safe profile |

## Latency (per stage, measured within detector node)

All numbers are **p50 / p95** over ≥1000 frames post 100-frame warmup.

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

### Real GO2 front-camera frames, per-frame 2D (2026-09-24)

The robot carries an RGB front camera only (960x540, 2 Hz, no depth), so the 3D path (back-projection, scene graph, Nav2 goal) cannot run on recorded data yet. This evaluates what can: detector, segmenter, encoder, grounding score and rejection gate, per frame, on real frames, with all models and thresholds frozen from `config/scene_profiles/dev_gpu.yaml`. Pre-registered before any compute in [`docs/preregistration/2026-09-24-real-frames-2d-grounding.md`](docs/preregistration/2026-09-24-real-frames-2d-grounding.md). Ground truth was committed before inference.

**Data.** 57 frames on a fixed 15 s grid from 3 recorded lab sessions. **One room, 9 distinct viewpoints** (the robot stood still for most of each recording). 682 queries: 209 present (chair, desk, table, person, bookshelf) and 473 absent (in-vocabulary classes not in view, plus guitar, dog, horse, bicycle, toilet). Present queries count as successes only if the chosen box actually contains the queried class; all 209 chosen boxes were judged by eye, shuffled and blind to the gate decision.

| Arm | Present success | Absent refusal | Balanced accuracy |
|---|---|---|---|
| Ungated (always take top-1) | 0.55 | 0.00 | 0.27 |
| Random refusal, dose-matched per session | 0.11 | 0.84 | 0.47 |
| **v2 gate** | **0.54** | **1.00** | **0.77** |
| Oracle (refuse iff truly absent) | 0.55 | 1.00 | 0.77 |

- **Primary (registered):** gate minus dose-matched sham = **+0.30**, 95% CI [0.25, 0.34] over 15 time blocks and [0.20, 0.37] over 9 viewpoint clusters. It clears the registered 0.15 bar. The sham's realised refusal rate matched the gate's within 0.001 in every session.
- **Every one of the 113 gate acceptances pointed at a correct box.** Misses are recall: the gate ties the oracle, so the bottleneck is detection and ranking (top-1 box correct 55%), not the gate.
- Per class, gate success: chair 0.93 (53/57), person 1.00 (32/32), desk 0.50 (28/56), **table 0.00 (0/56)**, bookshelf 0.00 (0/8).
- Latency on this data (dev-gpu, 960x540, 54 frames): total p50 83 ms / p95 136 ms; MobileSAM 51 / 77 ms, OpenCLIP 27 / 55 ms, YOLO-World 4.2 / 4.7 ms. **Not a Jetson number.**

**What this does not show.** (1) The gate reduced to a detector-label filter: the highest CLIP similarity of any top candidate was 0.273, below the 0.30 CLIP floor, so every acceptance came from an exact label match and the CLIP branch never decided anything. (2) The 100% absent refusal is not a stress test: no absent class was ever falsely detected under its own label here, which is the case the gate most needs to handle. (3) One room, one annotator (AI-assisted labelling from the frames, second-person check pending), per-frame only (no multi-frame scene-graph fusion), no navigation. Frames are not redistributed (people in view); rerun with `eval/real_frames/sample_frames.py`, `run_real_frames.py`, `analyze.py` against the source bags. Raw output: `eval/results/real_frames_2d/`.

Full 6-configuration ablation table lands in `eval/results/ablation.md`.

## Honest negative results

- **Raw-score thresholding is insufficient on small scenes (documented 2026-04-13).** The v1 grounding used a single `total_score >= 0.15` gate and accepted 16 out of 20 queries against a person-only synthetic scene, even for "approach the guitar" (chose person, total=0.179) and "the dog" (total=0.218). The underlying reason is that OpenCLIP ViT-B/16 gives raw image-text cosines in [0.1, 0.3] for clearly-unrelated pairs (CLIP text-image scores are not calibrated probabilities). The v2 two-layer gate (`absolute_floor=0.15 AND (label>=0.40 OR clip>=0.22)`) lifted expected-failure honesty from 0.00 → 0.67. One out-of-vocab query ("the dog") still slipped through because `CLIP("dog", person_crop) ≈ 0.23 > clip_floor`. A margin-based gate (top-1 − top-2) would catch it but reduces recall on scenes where multiple nearly-indistinguishable objects legitimately satisfy a query. The decision is deferred to Phase 5 post-rosbag eval.
- **"go to the table" fails on real frames (2026-09-24).** In our lab YOLO-World labels every table "desk", `lexical_label_match("table", "desk")` is 0.04, and the best CLIP-scored crop for "table" was a **monitor** in 50 of 56 frames. The gate refused all 56, so the robot would not have been sent to a monitor, but the query cannot succeed. Synonym handling (table/desk) or a lower CLIP floor are candidate fixes. Neither is tuned on these frames, since that would leak the evaluation set.
- **The 0.30 CLIP floor is above every real-frame similarity observed (max 0.273).** It was raised from 0.22 on a Gazebo recording. On real frames it makes the gate label-only. Lowering it needs fresh held-out real data, not these frames.
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
