# Contributing to go2-semantic-nav

Thanks for considering a contribution. This project is a research-grade ROS 2 overlay for a Unitree GO2, contributions need to preserve the overlay contract and respect the deploy target (Jetson Orin NX 16 GB at 25 W).

## Ground rules (non-negotiable)

1. **Do not edit the sibling `GO2-seeing-eye-dog` stack.** This repo overlays it. If your change requires something there, open an issue there first.
2. **Do not break interface contracts.** `/goal_pose` (`PoseStamped` in `map`), `/semantic/detections`, `/semantic/scene_graph`, `/semantic/query_objects`, `/semantic/ground_and_navigate`: their schema is frozen unless a major-version bump ships a migration doc.
3. **Match the QoS contract.** Camera image subs are `BEST_EFFORT` depth 1; camera_info is `RELIABLE` depth 10; `/goal_pose` is `RELIABLE` depth 10; semantic topics are `RELIABLE` depth 5.
4. **No live 3D Gaussian Splatting in the critical path.** The decision is documented in `docs/architecture.md` §"Why not 3DGS" and `decisions.md`. If you disagree, add an entry to `decisions.md` with measured evidence before landing code.
5. **Don't bump torch or cv_bridge numpy pins.** `numpy<2.0` is required for cv_bridge compat. Use a venv and pin.

## Repository layout: who owns what

| Area | Files | Ownership |
|---|---|---|
| Messages | `ros2_ws/src/go2_semantic_msgs/**` | frozen; touch only for explicit schema change |
| Detector node | `ros2_ws/src/go2_open_vocab_detector/**` | owns detection + mask + per-object embedding only |
| Scene graph | `ros2_ws/src/go2_scene_graph/**` | owns object tracking + spatial relations |
| Grounding | `ros2_ws/src/go2_language_grounding/**` | owns text-parse + scoring + goal sampling |
| Bringup | `ros2_ws/src/go2_semantic_bringup/**` | owns launch composition + RViz |
| Eval | `eval/**` | owns query sets + metrics + harness |
| Scripts | `scripts/**` | owns publisher, TRT export, benchmarks, demo |

Changes that span areas require a design note in `docs/` before implementation.

## Adding a new backend (detector / segmenter / encoder)

1. Drop a new module under `ros2_ws/src/go2_open_vocab_detector/go2_open_vocab_detector/backends/`.
2. Implement the `DetectorBackend`, `SegmenterBackend`, or `EncoderBackend` interface from `backends/base.py`.
3. Register it in `backends/factory.py` under the relevant `make_*()` with a stable string id.
4. Add latency rows to `RESULTS.md` for whichever platform(s) you measured on.
5. If the backend has Jetson-only deps (e.g., NanoSAM, SAM 2), keep imports lazy so dev hosts don't break.

## Development setup

```bash
# Clone + deps
git clone <repo>
cd go2-semantic-nav
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --symlink-install
source install/setup.bash

# Smoke test
python scripts/smoke_test_integration.py --text "person"
```

## Running checks

```bash
# Lint
ruff check ros2_ws/src eval scripts
black --check ros2_ws/src eval scripts

# Unit checks (bypass system pytest anyio issue by using pure logic)
python scripts/smoke_test_integration.py

# Eval (dev, synthetic)
python eval/run_eval.py --mode synthetic --warmup-s 15
```

## Commit convention

No `Co-Authored-By` lines for LLM tools. Short imperative subject:
```
feat(scene_graph): support dynamic label history
fix(grounding): normalize attribute scoring for single-word queries
docs(jetson): add NanoSAM engine path env var
```

## Pushing code

- Branch from `main`.
- Keep PRs scoped: one capability per PR, diff ≤ ~500 LOC where practical.
- Include a validation note: which check you ran, which hardware if relevant.
- Update `docs/` and `RESULTS.md` when your change affects them.
- Do NOT force-push shared branches.

## When you hit a build or env issue

Most common culprits (seen in this project's own history):

- **cv_bridge errors about numpy 2.x** → `pip install "numpy<2.0"`.
- **`ModuleNotFoundError: CLIP`** → `pip install git+https://github.com/openai/CLIP.git`, then immediately `pip install --no-deps --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"` if it clobbered your torch wheel.
- **CUDA arch mismatch on RTX 5070** → needs `torch` built for `sm_120` (CUDA 12.8). nvcc 11.5 cannot compile custom extensions; use ONNX + TensorRT instead.
- **Jetson `libnvinfer.so.10` missing** → reinstall `nvidia-tensorrt` for JetPack 6.x.

If none of those apply, open an issue with:
- OS + ROS distro + Python version + torch version + GPU model
- Output of `ros2 doctor --report`
- The exact command + the first ~50 lines of the error

## Questions?

Open an issue or email the maintainer. Please don't bring licensing questions into code reviews, handle those in a separate channel.
