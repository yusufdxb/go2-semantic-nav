#!/usr/bin/env python3
"""Record the camera node's 1 Hz latency windows, with the decoder clocks, for one clock A/B phase.

Used by `jetson_lab.sh clockab` (docs/preregistration/nvdec-clock-latency.md).
Subscribes only to /camera/front/latency, one small JSON message per second,
and not to the images: lab_probe.py also receives the 720p stream, and on the
robot computer deserialising that in Python competes for the CPU whose latency
is being measured.

Every window becomes one JSON line in --out, with each engine's devfreq
governor, min_freq and cur_freq read when the window arrives (the manipulation
check). Windows in the first --settle seconds are written with settle=true and
ignored by the analysis. Stops after --windows measured windows or --timeout
seconds, then prints a one-line summary.

Exit 0 when all --windows measured windows arrived, else 2.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def read_devfreq(path: str) -> dict:
    """Governor and current/min/max clock (Hz) of a devfreq node; {} without a path."""
    if not path:
        return {}
    out: dict = {}
    for name in ("governor", "cur_freq", "min_freq", "max_freq"):
        try:
            text = (Path(path) / name).read_text().strip()
        except OSError:
            out[name] = None
            continue
        out[name] = text if name == "governor" else int(text)
    return out


def window_record(stats: dict, *, t_s: float, settle_s: float, block: int, phase: str, order: str,
                  nvdec: dict, vic: dict) -> dict:
    """One latency window from the camera node's stats JSON, plus the clocks read with it."""
    lat = stats.get("arrival_to_publish_ms") or {}  # absent when no frame arrived that second
    rec = {"kind": "window", "block": block, "phase": phase, "order": order, "t_s": round(t_s, 3),
           "settle": t_s < settle_s, "fps": stats.get("fps"), "frames": stats.get("frames"),
           "restarts": stats.get("restarts"), "p50_ms": lat.get("p50"), "p95_ms": lat.get("p95"),
           "max_ms": lat.get("max")}
    for eng, clocks in (("nvdec", nvdec), ("vic", vic)):
        rec[f"{eng}_governor"] = clocks.get("governor")
        rec[f"{eng}_hz"] = clocks.get("cur_freq")
        rec[f"{eng}_min_hz"] = clocks.get("min_freq")
        rec[f"{eng}_max_hz"] = clocks.get("max_freq")
    return rec


def summarize(records: list[dict]) -> dict:
    m = [r for r in records if not r["settle"]]
    p50 = [r["p50_ms"] for r in m if r["p50_ms"] is not None]
    p95 = [r["p95_ms"] for r in m if r["p95_ms"] is not None]
    nv = [r["nvdec_hz"] for r in m if r["nvdec_hz"] is not None]
    at_max = [r["nvdec_hz"] == r["nvdec_max_hz"] for r in m if r["nvdec_hz"] is not None]
    first = records[0] if records else {}
    return {"block": first.get("block"), "phase": first.get("phase"), "windows": len(m),
            "settle_windows": len(records) - len(m),
            "p50_median_ms": round(statistics.median(p50), 2) if p50 else None,
            "p95_median_ms": round(statistics.median(p95), 2) if p95 else None,
            "nvdec_mhz_median": round(statistics.median(nv) / 1e6, 1) if nv else None,
            "nvdec_at_max": round(sum(at_max) / len(at_max), 2) if at_max else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--phase", default="-")
    ap.add_argument("--order", default="-")
    ap.add_argument("--settle", type=float, default=5.0, help="seconds of windows to mark settle=true")
    ap.add_argument("--windows", type=int, default=25, help="measured windows to collect")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--nvdec-devfreq", default="", help="devfreq folder of the NVDEC engine")
    ap.add_argument("--vic-devfreq", default="", help="devfreq folder of the VIC engine")
    ap.add_argument("--out", default="", help="JSON lines file to append to (none: summary only)")
    ap.add_argument("--topic", default="/camera/front/latency")
    a = ap.parse_args()

    import rclpy  # deferred: the record helpers above import without ROS
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("gsn_latency_window")
    records: list[dict] = []
    out = open(a.out, "a") if a.out else None
    t0 = time.monotonic()

    def on_stats(msg: String) -> None:
        try:
            stats = json.loads(msg.data)
        except ValueError:
            return
        rec = window_record(stats, t_s=time.monotonic() - t0, settle_s=a.settle, block=a.block, phase=a.phase,
                            order=a.order, nvdec=read_devfreq(a.nvdec_devfreq), vic=read_devfreq(a.vic_devfreq))
        records.append(rec)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()

    node.create_subscription(String, a.topic, on_stats, 10)
    try:
        while time.monotonic() - t0 < a.timeout and sum(not r["settle"] for r in records) < a.windows:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        if out:
            out.close()
    s = summarize(records)
    print(json.dumps(s))
    return 0 if s["windows"] >= a.windows else 2


if __name__ == "__main__":
    sys.exit(main())
