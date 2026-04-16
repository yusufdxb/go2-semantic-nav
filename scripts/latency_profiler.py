#!/usr/bin/env python3
"""Subscribe to /semantic/detections and report per-stage latency percentiles.

Uses the `latency_{detector,segmenter,encoder,backproject,total}_ms` fields
populated by `go2_open_vocab_detector`. Emits a JSON blob plus a concise
human-readable summary after N frames.

Run alongside a live or replayed semantic_nav stack:
    python scripts/latency_profiler.py --frames 1000 --out eval/results/latencies.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import rclpy
from go2_semantic_msgs.msg import SemanticDetectionArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

_FIELDS = (
    "latency_detector_ms",
    "latency_segmenter_ms",
    "latency_encoder_ms",
    "latency_backproject_ms",
    "latency_total_ms",
)


class LatencyCollector(Node):
    def __init__(self, target_frames: int) -> None:
        super().__init__("latency_collector")
        q = QoSProfile(depth=5)
        q.reliability = ReliabilityPolicy.RELIABLE
        self._sub = self.create_subscription(
            SemanticDetectionArray, "/semantic/detections", self._on, q
        )
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._target = target_frames
        self.done = False

    def _on(self, msg: SemanticDetectionArray) -> None:
        for f in _FIELDS:
            v = float(getattr(msg, f, -1.0))
            if v >= 0.0:
                self._samples[f].append(v)
        if len(self._samples["latency_total_ms"]) >= self._target:
            self.done = True

    def stats(self) -> dict:
        out: dict[str, dict] = {}
        for f in _FIELDS:
            xs = self._samples.get(f, [])
            if not xs:
                out[f] = {"n": 0}
                continue
            xs_sorted = sorted(xs)
            def pct(p: float) -> float:
                idx = int(p * (len(xs_sorted) - 1) / 100.0)
                return float(xs_sorted[idx])
            out[f] = {
                "n": len(xs),
                "mean": statistics.fmean(xs),
                "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
                "p50": pct(50),
                "p95": pct(95),
                "p99": pct(99),
                "min": xs_sorted[0],
                "max": xs_sorted[-1],
            }
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--warmup-frames", type=int, default=100)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--out", default="eval/results/latencies.json")
    args = ap.parse_args()

    rclpy.init()
    collector = LatencyCollector(target_frames=args.frames + args.warmup_frames)
    start = time.time()
    print(f"[latency] collecting {args.frames + args.warmup_frames} frames "
          f"(warmup {args.warmup_frames} discarded) ...")
    while not collector.done and time.time() - start < args.timeout_s:
        rclpy.spin_once(collector, timeout_sec=0.5)

    # Discard warmup
    for f in _FIELDS:
        if f in collector._samples and len(collector._samples[f]) > args.warmup_frames:
            collector._samples[f] = collector._samples[f][args.warmup_frames:]

    stats = collector.stats()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[latency] wrote {out_path}")
    # Pretty print
    n = stats["latency_total_ms"].get("n", 0)
    print(f"\n=== Per-stage latency over {n} frames ===")
    print(f"{'stage':22s}  p50    p95    p99    mean")
    for f in _FIELDS:
        s = stats[f]
        if s.get("n", 0) == 0:
            print(f"{f:22s}  (no samples)")
            continue
        print(f"{f:22s}  {s['p50']:5.1f}  {s['p95']:5.1f}  {s['p99']:5.1f}  {s['mean']:5.1f}  ms")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
