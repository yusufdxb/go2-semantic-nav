#!/usr/bin/env python3
"""Compare the C++ replay projection CSV with the original Python geometry."""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ros2_ws/src/go2_rgb_lidar"))


def frames(path):
    with Path(path).open("rb") as f:
        if f.read(8) != b"G2FUS001":
            raise ValueError("invalid replay magic")
        count = struct.unpack("<I", f.read(4))[0]
        if count > 100000:
            raise ValueError("frame capacity exceeded")
        for i in range(count):
            image_ns, cloud_ns, n = struct.unpack("<qqI", f.read(20))
            if n > 250000:
                raise ValueError("point capacity exceeded")
            transform = np.asarray(struct.unpack("<12d", f.read(96)))
            xyz = np.frombuffer(f.read(n * 12), dtype="<f4").reshape(n, 3)
            boxes = struct.unpack("<I", f.read(4))[0]
            if boxes > 64:
                raise ValueError("box capacity exceeded")
            detections = [struct.unpack("<5fI", f.read(24)) for _ in range(boxes)]
            yield i, image_ns, cloud_ns, xyz, transform, detections
        if f.read():
            raise ValueError("trailing replay data")


def main():
    from go2_rgb_lidar.calibration import load_camera_model
    from go2_rgb_lidar.projection import project_points, transform_points

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--frame", type=int, default=5)
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pixel-tolerance", type=float, default=0.001)
    a = p.parse_args()
    frame = next(x for x in frames(a.replay) if x[0] == a.frame)
    transform = np.eye(4)
    transform[:3, :3] = frame[4][:9].reshape(3, 3)
    transform[:3, 3] = frame[4][9:]
    cam = load_camera_model(a.intrinsics)
    pts = transform_points(transform, frame[3].astype(np.float64))
    u, v, z = project_points(pts, cam, 0.2, 8.0, 1.3)
    actual = np.genfromtxt(a.csv, delimiter=",", names=True)
    if len(actual) != len(u):
        raise ValueError(f"projected count mismatch C++={len(actual)}, Python={len(u)}")
    diff = np.column_stack([actual["u"] - u, actual["v"] - v, actual["z"] - z])
    # Visibility reference uses the same point footprints: no splat, float metric depths.
    ui = np.rint(u).astype(int)
    vi = np.rint(v).astype(int)
    depth = np.full((cam.height, cam.width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (vi, ui), z.astype(np.float32))
    import cv2

    near = cv2.erode(depth, np.ones((21, 21), np.uint8), borderType=cv2.BORDER_REPLICATE)
    expected_visible = (z <= near[vi, ui] + 0.3) & (z <= depth[vi, ui] + 1e-5)
    visible_mismatch = int(np.count_nonzero(expected_visible != actual["visible"].astype(bool)))
    report = {
        "frame": a.frame,
        "points": len(u),
        "max_abs_u_px": float(np.max(abs(diff[:, 0]))),
        "max_abs_v_px": float(np.max(abs(diff[:, 1]))),
        "max_abs_depth_m": float(np.max(abs(diff[:, 2]))),
        "visibility_mismatches": visible_mismatch,
        "pixel_tolerance": a.pixel_tolerance,
        "metric_accuracy_validated": False,
        "reference": "original Python float64 transform and projection; float metric depth OpenCV minimum visibility",
    }
    report["passed"] = (
        max(report["max_abs_u_px"], report["max_abs_v_px"]) <= a.pixel_tolerance
        and report["max_abs_depth_m"] <= 1e-5
        and visible_mismatch == 0
    )
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
