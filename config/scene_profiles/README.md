# Scene profiles

Pre-tuned backend + threshold combinations for the three deploy tiers documented in
`docs/architecture.md` §"Model backends".

| File | Platform | Intended use |
|---|---|---|
| `dev_mewtwo_rtx5070.yaml` | RTX 5070 workstation | Interactive development; biggest model we can afford at ~5 Hz |
| `jetson_tier_a.yaml` | Jetson Orin NX 25 W | **Default on-robot profile.** 3 Hz sustained with Nav2 headroom |
| `jetson_tier_b_burst.yaml` | Jetson Orin NX 25 W | Stationary room-scan bursts (higher recall, ~1 Hz) |
| `offboard_rtx_companion.yaml` | RTX 5070 laptop via DDS | Tier C — heavy stack offboard, robot subscribes |

## Usage

Pass the chosen profile via the composite launch file:

```bash
ros2 launch go2_semantic_bringup semantic_nav.launch.py \
    detector_params:=$(ros2 pkg prefix go2_semantic_bringup)/share/go2_semantic_bringup/config/jetson_tier_a.yaml \
    grounding_params:=$(ros2 pkg prefix go2_semantic_bringup)/share/go2_semantic_bringup/config/jetson_tier_a.yaml \
    scene_graph_params:=$(ros2 pkg prefix go2_semantic_bringup)/share/go2_semantic_bringup/config/jetson_tier_a.yaml
```

The launch file takes three separate params paths; pointing all three at the same
profile YAML applies a consistent tier across all nodes (YAML is keyed by node
name, so each node reads only its own section).

## How to add a profile

1. Copy the closest existing profile.
2. Change backend ids to match your target. Valid ids are enumerated in
   `ros2_ws/src/go2_open_vocab_detector/go2_open_vocab_detector/backends/factory.py`.
3. Tune the rejection floors after running `eval/run_eval.py` with the old
   thresholds against your scene — drop floors that refuse correct matches,
   raise floors that accept wrong matches. Document the measured ablation in
   `RESULTS.md`.
