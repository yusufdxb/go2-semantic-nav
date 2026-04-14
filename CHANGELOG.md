# Changelog

All notable changes to this project are documented here. Dates are `YYYY-MM-DD`.

This project adheres roughly to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) but without SemVer until after the first on-robot eval run.

## [Unreleased]

### Added
- 9 pluggable backends wired into the factory: YOLO-World v2 (s/m/l/x), YOLOE-11s, OWLv2 (base/large), Grounding-DINO (tiny/base), MobileSAM, NanoSAM, EfficientSAM (Tiny/Small), FastSAM-ready hook, SAM 2 (tiny/small/base+), OpenCLIP (B/16, L/14, H/14), MobileCLIP (S0/S1/S2/B), SigLIP (base/SO400M).
- Optional Qwen2-0.5B LLM parser fallback (`llm_parser.py`) for queries outside the rule grammar.
- Attribute-aware CLIP scoring via relative-prompt delta (`cos("red chair") − cos("chair")`), blended 60/40 with full similarity.
- Two-layer grounding rejection: absolute score floor AND combined label-floor / clip-floor — refuses to guess on weak matches instead of picking whatever's available.
- `scripts/synthetic_publisher.py`: hardware-free RGB-D publisher + static TF chain.
- `scripts/smoke_test_integration.py`: in-process end-to-end validation.
- `scripts/record_demo_bag.sh`: records the right topic set for offline eval replay.
- `scripts/latency_profiler.py`: per-stage latency percentiles from `/semantic/detections`.
- `scripts/run_thermal_benchmark.py`: 10-min thermal soak + sustained-Hz probe with `tegrastats` sampling (Jetson).
- TRT export scripts for YOLO-World (vocab-baked), OpenCLIP image encoder, MobileSAM (encoder+decoder).
- Jetson deploy container (`docker/jetson.Dockerfile` based on `l4t-pytorch:r36.2.0-pth2.3`).
- `slam_fallback.launch.py` + `use_slam_fallback` arg on composite bringup — covers the inherited `/odom` blocker.
- Real rosbag-backed eval harness (`eval/run_eval.py`) with synthetic + rosbag modes.
- `.github/workflows/ci.yml`: ruff lint + msgs build + pure-Python unit smoke.
- `CONTRIBUTING.md`, `RESULTS.md`, `docs/jetson_cookbook.md`, `docs/latency_instrumentation.md`, `docs/resume_bullets.md`, `docs/paper_outline.md`, `docs/github_summary.md`.

### Changed
- Raised grounding score threshold from a single `>0.15` gate to a two-layer rejection (absolute + label/clip floor) after the v1 eval surfaced false positives where the pipeline happily matched "guitar" to "person" at score ~0.18.
- `mobile_sam_segmenter.py` now resolves the checkpoint via a search path (`$checkpoint` arg → `~/.cache/mobile_sam/mobile_sam.pt` → cwd fallbacks) instead of requiring `mobile_sam.pt` in cwd.
- Cosine-similarity helpers in grounding + scene-graph no longer add `1e-9` to norms — which was polluting identity cases (`cosine(a,a)` returned 0.999... instead of 1.0). Zero-division guards now use `< 1e-9` checks on raw norms.

### Discovered (honest findings to document in the paper)
- Synthetic-scene eval on bus.jpg shows raw scoring is noise-dominated when the detected class does not match the query class: all 16 non-relation queries returned "person" at scores 0.15–0.22 before the two-layer gate was added. This is a legitimate failure mode to report, not hide.
- Ultralytics YOLO-World has a runtime dependency on OpenAI's `clip` that auto-installs via `uv pip install` at first inference. That installer can hang for minutes — pre-install manually in production deploys.
- `pip install --force-reinstall <non-torch-pkg>` without `--extra-index-url https://download.pytorch.org/whl/cu128` silently overwrites the Blackwell-compatible torch wheel with a default CUDA-13 one. Documented in `CONTRIBUTING.md` and `docs/troubleshooting.md`.
- System pytest has a stale anyio plugin that trips on `_pytest.scope` imports. Workaround: run logic directly via `python -c` or use a project-local venv.

## [0.1.0] — 2026-04-12
- Initial scaffold: 5 ROS 2 packages, Obsidian project registered, 7 memory seeds, all docs bootstrapped. All packages build; unit logic verified.
