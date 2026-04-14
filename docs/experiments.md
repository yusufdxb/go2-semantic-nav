# Experiments and evaluation

Goal: produce a reproducible, defensible evaluation that supports resume bullets and a paper submission.

## Metrics

| Metric | Definition | Target | Where measured |
|---|---|---|---|
| **Grounding top-1** | Fraction of queries where the highest-scored graph node is the operator-annotated correct object | ≥0.80 | eval script |
| **Grounding top-5** | Fraction of queries where the correct object is in the top-5 by score | ≥0.95 | eval script |
| **Navigation success (SR)** | Fraction of queries where the robot arrives within `stand_off_m + 0.3 m` of the true object and stops within 60 s | ≥0.65 | robot log + vicon or TF trace |
| **SPL** (Anderson et al. 2018) | Success weighted by Path Length: `SR × (shortest / max(shortest, actual))` | ≥0.45 | robot log |
| **End-to-end task success** | User says the query, robot arrives. Single binary human judgment, N judges | ≥0.65 | A/B labeling by two operators |
| **Grounding latency p50 / p95** | Action-goal-received → `/goal_pose`-published | p50 ≤ 0.5 s, p95 ≤ 1.2 s | rclpy timestamp |
| **Detection latency p50 / p95** | Image stamp → SemanticDetectionArray publish | p50 ≤ 250 ms, p95 ≤ 450 ms at Jetson 25 W | per-frame log |
| **Sustained detection rate** | Mean `/semantic/detections` rate over a 5-min run post thermal soak | ≥3 Hz at Jetson 25 W | ros2 topic hz |
| **Scene-graph object count** | #objects with `observation_count ≥ min_observations_to_publish` after 60 s of coverage | ≥10 in a typical office room | `/semantic/scene_graph` |
| **False-positive rate** | #graph objects that do not correspond to a real physical object, per minute of runtime | ≤1 / min | operator annotation |

## Query suite

`eval/queries.yaml` ships with 20 queries spanning:
- Direct object nouns ("chair", "table", "sofa")
- Attribute-qualified nouns ("red chair", "blue backpack")
- Unary spatial relations ("near the window", "in front of the television")
- Binary spatial relations ("the table next to the couch")
- Near-miss distractors (two similar objects with a disambiguating attribute)
- Out-of-prompt-list targets (e.g., "guitar" when the prompt set is indoor-generic) — these are expected failures; the interesting signal is whether the system returns `success=false` rather than a wrong goal.

Split: 10 queries for "seen" rooms (used in dev), 10 for "held-out" rooms.

## Scenes

- **Scene A** — mewtwo office (dev): 15 m², typical indoor objects, good lighting.
- **Scene B** — lab common area (dev): 40 m², mixed clutter.
- **Scene C** — held-out apartment living room: 25 m².

For the minimum-viable eval, Scene A + Scene C is sufficient.

## Reproducibility protocol

1. **Hardware declaration** — every reported number names the platform (RTX 5070 vs Jetson Orin NX 25 W) and power mode. Mixing is forbidden.
2. **Warmup** — discard first 100 detector frames; thermal soak 5 min on Jetson before measurement.
3. **Sample size** — ≥1000 frames for latency; ≥20 queries × 3 trials for grounding.
4. **Seed discipline** — pin numpy + torch seeds; for CLIP/YOLO models with deterministic inference, log the checksum.
5. **Raw data** — commit the per-frame CSV and per-query JSON alongside the analysis script.
6. **Version pinning** — declare ultralytics, open_clip_torch, torch, and CUDA versions in the result table.

## Reporting format

Per ablation, produce:

```csv
config, backend, encoder, segmenter, platform, power_mode,
  grounding_top1, grounding_top5, sr, spl, lat_grounding_p50_ms, lat_grounding_p95_ms,
  lat_detection_p50_ms, lat_detection_p95_ms, sustained_fps, notes
```

Plots:
- Bar chart: Grounding top-1 across 3 configs × 3 scenes
- Line chart: sustained FPS vs time over 10-min run (thermal plot on Jetson)
- Ablation table: {detector × encoder} heatmap for top-1 grounding

## Honest negative results to report

Include:
- At least one config where attribute disambiguation failed (e.g., "red chair" chose a different red object).
- At least one config where CLIP similarity ranked the wrong object higher despite correct label.
- Jetson p99 latency when sustained operation drifts into thermal throttling.

Hiding failures is a dishonest paper; reporting them is what makes the work credible.

## How to run

```bash
# From repo root
python eval/run_eval.py \
    --queries eval/queries.yaml \
    --scenes eval/scenes/office.bag,eval/scenes/living_room.bag \
    --configs eval/configs/jetson_tier_a.yaml,eval/configs/jetson_tier_b.yaml \
    --out eval/results/$(date +%Y%m%d_%H%M%S)
```

Outputs:
- `results.csv` with one row per (config, scene, query, trial)
- `summary.md` with aggregated metrics
- `latencies_per_frame.csv` when `--record-latency` is set
- `plots/` with matplotlib figures

## Paper outline seed

- **Title:** *Open-Vocabulary 3D Semantic Scene Graphs for Language-Grounded Quadruped Navigation on Edge Hardware*
- **Abstract (draft):** Open-vocabulary object-centric 3D semantic scene graphs enable real-time language-grounded goal generation for a Unitree GO2 quadruped on a Jetson Orin NX 16 GB. A YOLO-World + MobileSAM + CLIP pipeline populates a lazily-maintained graph in the `map` frame; a cosine-similarity grounding module with a lightweight spatial-relation resolver converts natural-language queries into reachable Nav2 goals. We report a 5-model ablation, a thermal-sustained latency profile at 25 W, and a 20-query evaluation across three rooms showing X% top-1 grounding and Y% end-to-end task success with explicit failure-mode analysis.
- **Venue candidates:** RSS workshop (late submission), ICRA, IROS, CoRL workshop, RA-L.
