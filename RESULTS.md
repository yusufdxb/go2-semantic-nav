# Results

Evaluation numbers for go2-semantic-nav. **Any claim in this file must be measured on the stated platform, not extrapolated.** Ranges in angle brackets are placeholders to be filled after Phase 5 eval runs on real hardware.

## Platforms evaluated

| Id | Description | GPU | CPU | RAM | Power | Notes |
|---|---|---|---|---|---|---|
| `mewtwo-5070` | Dev workstation | NVIDIA RTX 5070 (Blackwell, 12 GB) | Intel (host) | 32+ GB | ~250 W | Ubuntu 22.04, CUDA 12.8, torch 2.11+cu128 |
| `jetson-orin-nx-25w` | On-robot | NVIDIA Jetson Orin NX 16 GB (Ampere, 1024 CUDA cores) | 8-core ARM Cortex-A78AE | 16 GB unified | 25 W sustained (MAXN) | JetPack 6.x, TRT 10.x, thermal-soaked ≥5 min before measurement |
| `jetson-orin-nx-15w` | On-robot (power-capped) | same | same | same | 15 W sustained | thermal-safe profile |

## Latency (per stage, measured within detector node)

All numbers are **p50 / p95** over ≥1000 frames post 100-frame warmup.

| Config | Platform | Detector ms | Segmenter ms | Encoder ms | Backproject ms | Total ms |
|---|---|---|---|---|---|---|
| YOLO-Worldv2-s + MobileSAM + OpenCLIP-B/16 | **mewtwo-5070** | **4.6 / 4.8** | **37.7 / 38.6** | **20.7 / 21.3** | **4.6 / 5.0** | **67.7 / 69.0** |
| YOLO-Worldv2-s + MobileSAM + OpenCLIP-B/16 | jetson-orin-nx-25w | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |
| YOLO-Worldv2-s + NanoSAM + MobileCLIP-S2 | jetson-orin-nx-25w | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |
| YOLOE-11s + NanoSAM + MobileCLIP-S2 | jetson-orin-nx-25w | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |

**mewtwo-5070 measurement (2026-04-13, 50 frames post-10 warmup):** `eval/results/latencies_dev_mewtwo.json`. Segmenter dominates (MobileSAM encoder runs once per frame + 1 decode per box — 3 boxes here); YOLO-World detector at 4.6 ms is the cheapest stage. Total 68 ms → theoretical 14.7 Hz upper bound on RTX 5070 with this exact model choice. All latencies measured inside the detector node's callback via `time.perf_counter_ns()` and published in `SemanticDetectionArray.latency_*_ms`.

**Captured with the three-process pattern (2026-04-13):**
```bash
# term 1 (persistent):
python3 scripts/synthetic_publisher.py --ros-args -p rate_hz:=8.0
# term 2 (persistent):
ros2 launch go2_open_vocab_detector detector.launch.py
# term 3 (runs to completion):
python3 scripts/latency_profiler.py --frames 100 --warmup-frames 20 \
    --timeout-s 120 --out eval/results/latencies_dev_mewtwo.json
```
The SIGTERM-aware teardown in `synthetic_publisher.py` was added in the same pass to prevent `ExternalShutdownException` noise when processes are killed (e.g., by CI).

## Sustained throughput at Jetson 25 W

Post-thermal-soak (5 min), over 10-min continuous runs.

| Config | Detection Hz | Scene-graph Hz | GPU % | GPU memory MB | GPU temp °C |
|---|---|---|---|---|---|
| Tier-A (YOLO-Worldv2-s + NanoSAM + MobileCLIP-S2) | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` |

Thermal plot: `eval/results/latest/thermal_plot.png` (produced by `scripts/benchmark.py`).

## Grounding evaluation

Over 20 queries × 1 trial per config in `eval/queries.yaml` (dev-synthetic = bus.jpg only, hence no chair/table/sofa in scene — most in-vocab queries correctly get `success=False`).

| Config | Top-1 | Top-5 | Expected-failure honesty | Grounding p50 ms | Grounding p95 ms | Notes |
|---|---|---|---|---|---|---|
| dev-A-v1 (mewtwo-5070, loose thresholds) | 0.00 | 0.00 | 0.00 | 7.5 | 12.0 | baseline; system accepts every weak person-match |
| **dev-A-v2** (mewtwo-5070, tightened thresholds) | 0.00 | 0.00 | **0.67** | **7.8** | **21.3** | measured 2026-04-13 |
| jetson-A (Tier A) | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` | needs hardware |
| jetson-B (Tier B burst) | `<…>` | `<…>` | `<…>` | `<…>` | `<…>` | needs hardware |

**dev-A-v2 raw output:** `eval/results/dev_mewtwo_synthetic_v2/summary.json`.

**Note on dev top-1:** 0.00 is the correct result for the synthetic scene — bus.jpg contains only people, but the 20-query suite targets chairs, windows, tables, sofas. A 0 top-1 score for "chair" when no chair is physically in the scene is honest behavior, not a regression; the honesty metric is what's informative here. A real indoor rosbag is needed to measure meaningful top-1 / top-5.

## End-to-end navigation

Measured on GO2 EDU with `seeing-eye-dog` Nav2 downstream; ≥3 trials per query.

| Config | Navigation SR | SPL | Mean final-goal error m |
|---|---|---|---|
| Tier-A on robot | `<…>` | `<…>` | `<…>` |

## Ablation

Full 6-configuration ablation table lands in `eval/results/ablation.md`.

## Honest negative results

- **Raw-score thresholding is insufficient on small scenes (documented 2026-04-13).** The v1 grounding used a single `total_score >= 0.15` gate and accepted 16 out of 20 queries against a person-only synthetic scene — even for "approach the guitar" (chose person, total=0.179) and "the dog" (total=0.218). The underlying reason is that OpenCLIP ViT-B/16 gives raw image-text cosines in [0.1, 0.3] for clearly-unrelated pairs (CLIP text-image scores are not calibrated probabilities). The v2 two-layer gate (`absolute_floor=0.15 AND (label>=0.40 OR clip>=0.22)`) lifted expected-failure honesty from 0.00 → 0.67. One out-of-vocab query ("the dog") still slipped through because `CLIP("dog", person_crop) ≈ 0.23 > clip_floor`. A margin-based gate (top-1 − top-2) would catch it but reduces recall on scenes where multiple nearly-indistinguishable objects legitimately satisfy a query. The decision is deferred to Phase 5 post-rosbag eval.
- **Ultralytics YOLO-World has an on-demand CLIP auto-install path** that is not declared in its package metadata. First-run inference triggers `uv pip install git+https://github.com/ultralytics/CLIP.git` which can hang for minutes and does not respect offline environments. Workaround documented in `docs/troubleshooting.md`.
- **Raw `pip install --force-reinstall`** without `--extra-index-url https://download.pytorch.org/whl/cu128` silently replaces the Blackwell-compatible `torch==2.11.0+cu128` wheel with a default CUDA-13 one that fails to run on the RTX 5070 driver (12.8). Documented in `CONTRIBUTING.md`.

## Reproducibility

All numbers above are reproducible via:

```bash
# dev-A-v2 run (mewtwo-5070 synthetic)
cd go2-semantic-nav
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
python eval/run_eval.py --mode synthetic --warmup-s 15 \
    --out eval/results/dev_mewtwo_synthetic_v2 \
    --config-name dev_mewtwo_synthetic_v2

# Jetson run (needs hardware)
# [Jetson] after launching semantic_nav bringup against real camera:
python3 eval/run_eval.py --mode rosbag --bag /data/indoor_a.bag \
    --out eval/results/jetson_A --config-name jetson_A
```

Pinned versions: see `requirements.txt` and `docker/{dev,jetson}.Dockerfile`. Hardware spec: this file. Power mode and thermal state: this file.
