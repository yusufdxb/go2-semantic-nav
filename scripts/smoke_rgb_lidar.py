#!/usr/bin/env python3
"""Smoke test: real GO2 camera frame -> lidar_depth_node -> real detector -> 3D detections.

The image is a real frame from a recorded GO2 bag. The LiDAR cloud is SYNTHETIC
(no bag holds the front camera and the LiDAR together): a wall 2.0 m ahead on
the left half of the view and 5.0 m ahead on the right half. Every detection
must therefore come back with the depth of the half its mask lies in, and a
camera-frame x consistent with its pixel position. That checks the plumbing
the robot will exercise (message sync, alignment gate, depth encoding, camera
info, the TF chain, min-valid-pixel gating) with the real detector models; it
says nothing about calibration accuracy or real LiDAR density.

    python3 scripts/smoke_rgb_lidar.py --bag <bag with /camera/front/image_raw> --frame 100
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

NEAR_M, FAR_M = 2.0, 5.0
CAM_FRAME = "front_camera_optical_frame"
EXT_XYZ = [0.32, 0.0, 0.03]
EXT_RPY = [-math.pi / 2, 0.0, -math.pi / 2]


def read_frame(bag: str, index: int) -> Image:
    import rosbag2_py
    from rclpy.serialization import deserialize_message

    storage = "mcap" if any(f.endswith(".mcap") for f in os.listdir(bag)) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/camera/front/image_raw"]))
    i = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        if i == index:
            return deserialize_message(data, Image)
        i += 1
    raise SystemExit(f"bag has only {i} front-camera frames")


def write_calibration(tmp: str, width: int, height: int) -> tuple[str, str, float]:
    fx = (width / 2) / math.tan(math.radians(50))  # documented 100 deg HFOV
    intr = {
        "image_width": width,
        "image_height": height,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [fx, 0, width / 2, 0, fx, height / 2, 0, 0, 1]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0, 0, 0, 0, 0]},
    }
    ext = {"parent_frame": "base_link", "child_frame": CAM_FRAME, "xyz": EXT_XYZ, "rpy": EXT_RPY}
    ip, ep = os.path.join(tmp, "intr.yaml"), os.path.join(tmp, "ext.yaml")
    with open(ip, "w") as f:
        yaml.safe_dump(intr, f)
    with open(ep, "w") as f:
        yaml.safe_dump(ext, f)
    return ip, ep, fx


def synthetic_walls_in_odom() -> np.ndarray:
    """Camera-frame step scene (near left, far right) expressed in odom (= base_link here)."""
    from go2_rgb_lidar.projection import make_transform, rpy_to_matrix, transform_points

    near = [[x, y, NEAR_M] for x in np.arange(-3.0, -0.05, 0.04) for y in np.arange(-2.0, 2.0, 0.04)]
    far = [[x, y, FAR_M] for x in np.arange(0.05, 7.0, 0.08) for y in np.arange(-4.0, 4.0, 0.08)]
    pts_cam = np.array(near + far)
    base_from_cam = make_transform(rpy_to_matrix(*EXT_RPY), EXT_XYZ)
    return transform_points(base_from_cam, pts_cam).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory with /camera/front/image_raw")
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    frame = read_frame(args.bag, args.frame)
    frame.header.frame_id = CAM_FRAME
    tmp = tempfile.mkdtemp(prefix="smoke_rgb_lidar_")
    ip, ep, fx = write_calibration(tmp, frame.width, frame.height)
    cx = frame.width / 2
    cloud_odom = synthetic_walls_in_odom()

    rclpy.init()
    from go2_open_vocab_detector.detector_node import DetectorNode
    from go2_rgb_lidar.lidar_depth_node import LidarDepthNode
    from go2_semantic_msgs.msg import SemanticDetectionArray

    depth_node = LidarDepthNode(parameter_overrides=[
        Parameter("intrinsics_file", value=ip),
        Parameter("extrinsics_file", value=ep),
        Parameter("max_rate_hz", value=0.0),
    ])
    share = os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "go2_open_vocab_detector", "config")
    with open(os.path.join(share, "detector_rgb_lidar.yaml")) as f:
        det_params = yaml.safe_load(f)["go2_open_vocab_detector"]["ros__parameters"]
    det_params["device"] = args.device
    det_params["detection_rate_hz"] = 2.0
    detector = DetectorNode(parameter_overrides=[Parameter(k, value=v) for k, v in det_params.items()])

    io = rclpy.create_node("smoke_rgb_lidar_io")
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "odom", "base_link"
    tf.transform.rotation.w = 1.0
    static = StaticTransformBroadcaster(io)
    static.sendTransform(tf)
    image_pub = io.create_publisher(Image, "/camera/front/image_raw", qos_profile_sensor_data)
    cloud_pub = io.create_publisher(point_cloud2.PointCloud2, "/go2/lidar/points", qos_profile_sensor_data)
    results: list = []
    io.create_subscription(SemanticDetectionArray, "/semantic/detections", results.append, 10)

    def tick():
        now = io.get_clock().now().to_msg()
        cloud_pub.publish(point_cloud2.create_cloud_xyz32(Header(frame_id="odom", stamp=now), cloud_odom))
        frame.header.stamp = now
        image_pub.publish(frame)

    io.create_timer(0.5, tick)
    ex = MultiThreadedExecutor()
    for n in (depth_node, detector, io):
        ex.add_node(n)

    print(f"[smoke] frame {args.frame}: {frame.width}x{frame.height}, cloud {len(cloud_odom)} pts; waiting for detections ...")
    end = time.monotonic() + args.timeout
    while time.monotonic() < end and not any(len(r.detections) for r in results):
        ex.spin_once(timeout_sec=0.05)

    ok = False
    found = [r for r in results if len(r.detections)]
    if not found:
        print(f"[smoke] FAIL: no detections in {args.timeout:.0f} s (depth published={depth_node.stats})")
    else:
        msg = found[0]
        print(f"[smoke] depth node stats={depth_node.stats}, valid depth pixels={depth_node.last_valid_pixels}")
        ok = True
        for d in msg.detections:
            u = d.bbox_2d.center.position.x
            z = d.depth_median_m
            x = d.centroid_3d_camera.x
            mask_side = "left" if d.mask_roi.x_offset + d.mask_roi.width <= cx else (
                "right" if d.mask_roi.x_offset >= cx else "both")
            expect = {"left": NEAR_M, "right": FAR_M}.get(mask_side)
            x_expect = (u - cx) * z / fx
            good = (expect is None or abs(z - expect) < 0.05) and abs(x - x_expect) < 0.15 * z
            ok &= good
            print(f"[smoke] {'ok  ' if good else 'BAD '} {d.label:<14} score={d.score:.2f} bbox_u={u:6.1f} "
                  f"mask={mask_side:<5} depth={z:.2f} m (expect {expect}) x={x:+.2f} m (pixel-implied {x_expect:+.2f})")
        print(f"[smoke] {'PASS' if ok else 'FAIL'}: {len(msg.detections)} detections in frame {msg.header.frame_id!r}")

    ex.shutdown()
    for n in (depth_node, detector, io):
        n.destroy_node()
    rclpy.try_shutdown()
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
