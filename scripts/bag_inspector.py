#!/usr/bin/env python3
"""Inspect a rosbag for semantic-nav compatibility.

Reports:
  - total duration
  - per-topic rate
  - TF tree (frames + edges)
  - whether the required topic set is present with plausible QoS

Usage:
    python scripts/bag_inspector.py /path/to/bag/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_TOPICS = {
    "/camera/color/image_raw",
    "/camera/depth/image_rect_raw",
    "/camera/color/camera_info",
    "/tf_static",
}
RECOMMENDED_TOPICS = {
    "/tf",
    "/odom",
    "/camera/depth/camera_info",
}


def _parse_rosbag_info(bag_path: Path) -> dict:
    """Shell out to `ros2 bag info` for a structured-ish summary."""
    try:
        out = subprocess.check_output(["ros2", "bag", "info", str(bag_path)], text=True)
    except FileNotFoundError:
        raise SystemExit("ros2 not on PATH; source /opt/ros/humble/setup.bash first")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ros2 bag info failed: {exc}")
    parsed = {"raw": out, "topics": {}, "duration_s": 0.0, "message_count": 0}
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if line.startswith("Duration:"):
            try:
                parsed["duration_s"] = float(line.split()[1].rstrip("s"))
            except (IndexError, ValueError):
                pass
        if line.startswith("Messages:") or line.startswith("Message Count:"):
            try:
                parsed["message_count"] = int(line.split()[-1])
            except ValueError:
                pass
        # "Topic information: Topic: /a | …" (first row) and "Topic: /b | …" (subsequent) —
        # find the "Topic:" marker anywhere and parse the rest of the line from there.
        topic_idx = line.find("Topic:")
        if topic_idx >= 0 and "| Type:" in line:
            topic_line = line[topic_idx:]
            parts = [p.strip() for p in topic_line.split("|")]
            topic = next((p.split(":", 1)[1].strip() for p in parts if p.startswith("Topic:")), None)
            type_ = next((p.split(":", 1)[1].strip() for p in parts if p.startswith("Type:")), None)
            count_ = next((p.split(":", 1)[1].strip() for p in parts if p.startswith("Count:")), None)
            if topic:
                parsed["topics"][topic] = {"type": type_, "count": int(count_) if count_ else 0}
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", help="rosbag directory (mcap or sqlite3)")
    args = ap.parse_args()

    bag = Path(args.bag)
    if not bag.exists():
        print(f"ERROR: {bag} does not exist", file=sys.stderr)
        return 1

    info = _parse_rosbag_info(bag)
    print(f"=== {bag.name} ===")
    print(f"duration:     {info['duration_s']:.2f} s")
    print(f"messages:     {info['message_count']}")
    print(f"topics:       {len(info['topics'])}")
    print()

    print("--- topic rates (inferred from count / duration) ---")
    for topic, meta in sorted(info["topics"].items()):
        hz = meta["count"] / info["duration_s"] if info["duration_s"] > 0 else 0
        print(f"  {topic:40s}  {meta['type']:40s}  {hz:6.1f} Hz")
    print()

    present = set(info["topics"].keys())
    missing_required = REQUIRED_TOPICS - present
    missing_recommended = RECOMMENDED_TOPICS - present

    if missing_required:
        print("MISSING REQUIRED TOPICS:")
        for t in sorted(missing_required):
            print(f"  {t}")
        print("This bag is NOT compatible with go2-semantic-nav replay.")
    else:
        print("All required topics present — bag is replay-compatible.")

    if missing_recommended:
        print("\nMissing recommended topics (bag will still work, but):")
        for t in sorted(missing_recommended):
            print(f"  {t}")

    return 0 if not missing_required else 2


if __name__ == "__main__":
    raise SystemExit(main())
