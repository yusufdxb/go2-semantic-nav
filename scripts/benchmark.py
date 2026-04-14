#!/usr/bin/env python3
"""Benchmark detector + scene graph + grounding latencies from recorded rosbags.

Produces a per-frame CSV and a summary. Does NOT drive a robot. Intended for
the Phase 5 ablation study.

Usage:
    python scripts/benchmark.py --bag data/indoor.bag --config configs/jetson_tier_a.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag path")
    ap.add_argument("--config", required=True, help="YAML config identifying the backend stack")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default="eval/results/latest")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "latencies_per_frame.csv"
    summary_path = out_dir / "summary.json"

    print(f"[benchmark] bag={args.bag} config={args.config} frames={args.frames}")
    print(f"[benchmark] (stub) replace me with a rosbag2_py reader loop that subscribes to")
    print(f"[benchmark] /semantic/detections and logs latency_* fields per message.")

    # Scaffold: write a trivial header so downstream analysis scripts have a stable shape.
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx",
            "stamp_ns",
            "latency_detector_ms",
            "latency_segmenter_ms",
            "latency_encoder_ms",
            "latency_backproject_ms",
            "latency_total_ms",
            "n_detections",
        ])

    summary = {
        "bag": args.bag,
        "config": args.config,
        "frames_requested": args.frames,
        "frames_written": 0,
        "note": "stub — implement rosbag2_py driver in phase 5",
        "wall_time_s": 0.0,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[benchmark] wrote stub results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
