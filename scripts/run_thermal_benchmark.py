#!/usr/bin/env python3
"""Thermal-soak + sustained-rate benchmark for go2-semantic-nav on Jetson.

Runs for `--duration-s` (default 600 s = 10 min) at a fixed detector rate,
samples tegrastats at 1 Hz, measures `/semantic/detections` Hz via a subscriber,
and emits `thermal_plot.png` + `thermal.csv` for the RESULTS.md thermal row.

Must be run on the Jetson after `sudo nvpmodel -m 0 && sudo jetson_clocks`.
Assumes semantic_nav.launch.py is already running. Does not launch nodes itself.
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from go2_semantic_msgs.msg import SemanticDetectionArray


class HzProbe(Node):
    def __init__(self) -> None:
        super().__init__("thermal_hz_probe")
        q = QoSProfile(depth=5)
        q.reliability = ReliabilityPolicy.RELIABLE
        self._sub = self.create_subscription(
            SemanticDetectionArray, "/semantic/detections", self._on, q
        )
        self.last_ns = 0
        self.samples: list[float] = []  # inter-arrival seconds

    def _on(self, _msg) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self.last_ns:
            dt = (now_ns - self.last_ns) / 1e9
            if 0.0 < dt < 5.0:
                self.samples.append(dt)
        self.last_ns = now_ns


_TEGRA_LINE = re.compile(
    r"GR3D_FREQ (?P<gpu>\d+)%.*?"
    r"(CPU@(?P<cpu_temp>[\d\.]+)C)?"
    r".*?(GPU@(?P<gpu_temp>[\d\.]+)C)?"
    r".*?(VDD_IN (?P<vdd_in>\d+)mW)?",
    re.VERBOSE,
)


def _parse_tegrastats_line(line: str) -> dict:
    """Best-effort parse of a tegrastats line. Unknown fields become None."""
    d = {"raw": line.strip()}
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    d["gpu_util_pct"] = int(m.group(1)) if m else None
    m = re.search(r"CPU@([\d\.]+)C", line)
    d["cpu_temp_c"] = float(m.group(1)) if m else None
    m = re.search(r"GPU@([\d\.]+)C", line)
    d["gpu_temp_c"] = float(m.group(1)) if m else None
    m = re.search(r"VDD_IN (\d+)mW", line)
    d["vdd_in_mw"] = int(m.group(1)) if m else None
    m = re.search(r"VDD_CPU_GPU_CV (\d+)mW", line)
    d["vdd_cpu_gpu_cv_mw"] = int(m.group(1)) if m else None
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    d["ram_used_mb"] = int(m.group(1)) if m else None
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration-s", type=float, default=600.0)
    ap.add_argument("--sample-interval-ms", type=int, default=1000)
    ap.add_argument("--out", default="eval/results/thermal")
    ap.add_argument("--tegrastats", default="/usr/bin/tegrastats")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "thermal.csv"

    rclpy.init()
    probe = HzProbe()

    tegra_proc: Optional[subprocess.Popen] = None
    try:
        tegra_proc = subprocess.Popen(
            [args.tegrastats, "--interval", str(args.sample_interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        print(f"[thermal] tegrastats not found at {args.tegrastats}. On dev hosts this is expected.", file=sys.stderr)

    start = time.time()
    rows: list[dict] = []
    print(f"[thermal] running for {args.duration_s:.0f} s; sampling every {args.sample_interval_ms} ms")

    def _stop(signum, frame):
        if tegra_proc is not None:
            tegra_proc.terminate()

    signal.signal(signal.SIGINT, _stop)

    while time.time() - start < args.duration_s:
        rclpy.spin_once(probe, timeout_sec=0.2)
        if tegra_proc is not None and tegra_proc.stdout is not None:
            line = tegra_proc.stdout.readline()
            if line:
                parsed = _parse_tegrastats_line(line)
                parsed["elapsed_s"] = time.time() - start
                if probe.samples:
                    recent = probe.samples[-10:]
                    parsed["detection_hz_recent"] = 1.0 / (sum(recent) / len(recent)) if recent else 0.0
                else:
                    parsed["detection_hz_recent"] = 0.0
                rows.append(parsed)

    if tegra_proc is not None:
        tegra_proc.terminate()
        try:
            tegra_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            tegra_proc.kill()

    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # Summary
        hz_samples = [r["detection_hz_recent"] for r in rows if r.get("detection_hz_recent", 0) > 0]
        gpu_temps = [r["gpu_temp_c"] for r in rows if r.get("gpu_temp_c") is not None]
        gpu_utils = [r["gpu_util_pct"] for r in rows if r.get("gpu_util_pct") is not None]
        power = [r["vdd_in_mw"] for r in rows if r.get("vdd_in_mw") is not None]

        def _stats(xs, unit=""):
            if not xs:
                return "n/a"
            return f"mean={sum(xs)/len(xs):.1f}{unit} min={min(xs):.1f}{unit} max={max(xs):.1f}{unit}"

        print(f"\n[thermal] {len(rows)} samples over {args.duration_s:.0f} s")
        print(f"  detection Hz: {_stats(hz_samples, ' Hz')}")
        print(f"  GPU temp:     {_stats(gpu_temps, ' C')}")
        print(f"  GPU util:     {_stats(gpu_utils, ' %')}")
        print(f"  power in:     {_stats(power, ' mW')}")
        print(f"  wrote: {csv_path}")
    else:
        print("[thermal] no samples collected (tegrastats missing?)")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
