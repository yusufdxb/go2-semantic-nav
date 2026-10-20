#!/usr/bin/env python3
"""Live health and latency probe for the RGB + LiDAR path, run on the robot computer.

Listens for --seconds and reports, with PASS/FAIL against the limits below:

  camera      frames/s received here, image size, encoding, frame_id
  camera node arrival->publish p50/p95 (ms) and frames published, from /camera/front/latency
  delivery    frames received here / frames the camera node published (DDS loss)
  lidar       relayed clouds/s and points per cloud
  odom        relayed odometry/s
  depth       lidar_depth images/s and valid pixels per image (needs a calibration,
              or allow_nominal_calibration:=true)

Prints one JSON line (for the run log) after the table. Exit 0 = all PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

LIMITS = {
    "camera_fps_min": 12.0,          # GO2 stream is documented at 15 fps
    "arrival_to_publish_p95_ms_max": 25.0,
    "delivery_min": 0.95,
    "lidar_hz_min": 10.0,            # 15.4 Hz measured on this robot
    "odom_hz_min": 50.0,             # ~150 Hz measured
    "depth_hz_min": 3.0,             # lidar_depth max_rate_hz 5
    "depth_valid_px_min": 2000,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--expect-width", type=int, default=0)
    ap.add_argument("--expect-height", type=int, default=0)
    ap.add_argument("--no-depth", action="store_true", help="skip the depth checks")
    a = ap.parse_args()

    rclpy.init()
    n = rclpy.create_node("rgb_lidar_lab_probe")
    s = {"img": 0, "img_t": [], "cam_t": [], "size": None, "enc": None, "frame": None, "cloud": 0, "pts": [], "odom": 0, "depth": 0,
         "valid": [], "cam": []}

    def on_img(m: Image):
        s["img"] += 1
        s["img_t"].append(time.monotonic())
        s["size"], s["enc"], s["frame"] = (m.width, m.height), m.encoding, m.header.frame_id

    def on_depth(m: Image):
        s["depth"] += 1
        if s["depth"] % 3 == 1:  # sample: counting 1M pixels per frame in Python is not free
            s["valid"].append(int(np.count_nonzero(np.frombuffer(m.data, np.uint16))))

    def on_cloud(m: PointCloud2):
        s["cloud"] += 1
        s["pts"].append(m.width * m.height)

    n.create_subscription(Image, "/camera/front/image_raw", on_img, qos_profile_sensor_data)
    n.create_subscription(Image, "/camera/front/lidar_depth", on_depth, qos_profile_sensor_data)
    n.create_subscription(PointCloud2, "/go2/lidar/points", on_cloud, qos_profile_sensor_data)
    n.create_subscription(Odometry, "/odom", lambda m: s.__setitem__("odom", s["odom"] + 1), 50)
    def on_cam(m: String):
        s["cam"].append(json.loads(m.data))
        s["cam_t"].append(time.monotonic())

    n.create_subscription(String, "/camera/front/latency", on_cam, 10)

    t0 = time.monotonic()
    while time.monotonic() - t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
    dt = time.monotonic() - t0
    n.destroy_node()
    rclpy.try_shutdown()

    cam = s["cam"]
    # Delivery over the same window: frames the node counted between its first
    # and last stats message vs frames that arrived here in that interval.
    published = (cam[-1]["frames"] - cam[0]["frames"]) if len(cam) >= 2 else 0
    window_s = (s["cam_t"][-1] - s["cam_t"][0]) if len(cam) >= 2 else 1.0
    received_in_window = sum(1 for t in s["img_t"] if s["cam_t"][0] < t <= s["cam_t"][-1]) if len(cam) >= 2 else 0
    p95s = [c["arrival_to_publish_ms"]["p95"] for c in cam if "arrival_to_publish_ms" in c]
    p50s = [c["arrival_to_publish_ms"]["p50"] for c in cam if "arrival_to_publish_ms" in c]
    r = {
        "camera_fps": s["img"] / dt,
        "image_size": s["size"],
        "encoding": s["enc"],
        "frame_id": s["frame"],
        "node_published_fps": published / window_s if published else 0.0,
        "arrival_to_publish_p50_ms": float(np.median(p50s)) if p50s else None,
        "arrival_to_publish_p95_ms": float(np.max(p95s)) if p95s else None,
        "restarts": cam[-1]["restarts"] if cam else None,
        "delivery": min(1.0, received_in_window / published) if published else 0.0,
        "lidar_hz": s["cloud"] / dt,
        "lidar_points_median": int(np.median(s["pts"])) if s["pts"] else 0,
        "odom_hz": s["odom"] / dt,
        "depth_hz": s["depth"] / dt,
        "depth_valid_px_median": int(np.median(s["valid"])) if s["valid"] else 0,
    }
    checks = [
        ("camera fps", r["camera_fps"] >= LIMITS["camera_fps_min"], f"{r['camera_fps']:.1f}"),
        ("arrival->publish p95 ms", r["arrival_to_publish_p95_ms"] is not None
         and r["arrival_to_publish_p95_ms"] <= LIMITS["arrival_to_publish_p95_ms_max"], str(r["arrival_to_publish_p95_ms"])),
        ("DDS delivery", r["delivery"] >= LIMITS["delivery_min"], f"{r['delivery']:.3f}"),
        ("lidar Hz", r["lidar_hz"] >= LIMITS["lidar_hz_min"], f"{r['lidar_hz']:.1f} ({r['lidar_points_median']} pts)"),
        ("odom Hz", r["odom_hz"] >= LIMITS["odom_hz_min"], f"{r['odom_hz']:.0f}"),
    ]
    if a.expect_width:
        checks.append(("image size", r["image_size"] == (a.expect_width, a.expect_height), str(r["image_size"])))
    if not a.no_depth:
        checks.append(("depth Hz", r["depth_hz"] >= LIMITS["depth_hz_min"], f"{r['depth_hz']:.1f}"))
        checks.append(("depth valid px", r["depth_valid_px_median"] >= LIMITS["depth_valid_px_min"],
                       str(r["depth_valid_px_median"])))
    for name, ok, val in checks:
        print(f"[probe] {'PASS' if ok else 'FAIL'}  {name:<24} {val}")
    r["pass"] = all(ok for _, ok, _ in checks)
    print(json.dumps(r))
    return 0 if r["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
