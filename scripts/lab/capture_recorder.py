#!/usr/bin/env python3
"""Record one capture segment on the robot for the offline calibration tools (scripts/calib).

Writes <out>/<segment>/{meta.yaml, frames.csv, frames/*.jpg, clouds.csv,
clouds/*.npy, odom.csv} (format: scripts/calib/README.md) and prints a status
line every 2 s so the operator knows when the segment is good enough:

  checkerboard : boards found, 4x4 image coverage map; stops early once
                 --min-views boards cover >= 75 % of the grid
  scene        : frames, clouds, and LiDAR points per cloud
  yaw          : current base yaw rate from odometry

    capture_recorder.py --out ~/captures/<run> --segment checkerboard --seconds 120 \
        --intrinsics <yaml> --extrinsics <yaml> --latency-s 0.0

Records only: subscribes to the camera, the relayed LiDAR and odometry, and
never publishes.
"""

from __future__ import annotations

import argparse
import datetime
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


def _ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _bgr(msg: Image) -> np.ndarray:
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"unsupported encoding {msg.encoding}")
    if len(msg.data) != msg.height * msg.step or msg.step < msg.width * 3:
        raise ValueError(f"{msg.width}x{msg.height} step {msg.step} does not match {len(msg.data)} bytes")
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = img.reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1] if msg.encoding == "rgb8" else img


def parse_taped(items: list[str]) -> list[dict]:
    """'chair:1.5:0.0' or 'chair:2.0:-0.8:0.3' -> {label, x_m, y_m[, z_m]} in base_link."""
    out = []
    for it in items:
        parts = it.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"taped object {it!r}: want label:x:y[:z]")
        d = {"label": parts[0], "x_m": float(parts[1]), "y_m": float(parts[2])}
        if len(parts) == 4:
            d["z_m"] = float(parts[3])
        out.append(d)
    return out


class Recorder:
    def __init__(self, node, a) -> None:
        self.a = a
        self.dir = os.path.join(os.path.expanduser(a.out), a.segment)
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "clouds"), exist_ok=True)
        self.frames_csv = open(os.path.join(self.dir, "frames.csv"), "w")
        self.frames_csv.write("stamp_ns,file,width,height\n")
        self.clouds_csv = open(os.path.join(self.dir, "clouds.csv"), "w")
        self.clouds_csv.write("stamp_ns,file,frame_id,n_points\n")
        self.odom_csv = open(os.path.join(self.dir, "odom.csv"), "w")
        self.odom_csv.write("stamp_ns,x,y,z,qx,qy,qz,qw,wz\n")
        self.lock = threading.Lock()
        self.jobs: queue.Queue = queue.Queue(maxsize=64)
        self.n_frames = self.n_clouds = self.n_odom = self.n_dropped = self.n_bad = 0
        self.last_points = 0
        self.last_wz = 0.0
        self.last_save = 0.0
        self.boards = 0
        self.grid = np.zeros((4, 4), dtype=int)
        self.size = None
        self.board = tuple(int(v) for v in a.board.split("x"))
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()
        node.create_subscription(Image, a.image_topic, self.on_image, qos_profile_sensor_data)
        node.create_subscription(PointCloud2, a.cloud_topic, self.on_cloud, qos_profile_sensor_data)
        node.create_subscription(Odometry, a.odom_topic, self.on_odom, 50)

    # Image encoding and board detection run off the executor thread so
    # clouds and odometry keep flowing; if the writer falls behind, frames are
    # dropped and counted rather than queued without bound.
    def on_image(self, msg: Image) -> None:
        now = time.monotonic()
        if self.a.image_rate > 0 and now - self.last_save < 1.0 / self.a.image_rate:
            return
        self.last_save = now
        try:
            img = _bgr(msg).copy()
        except ValueError as exc:  # a malformed frame must not end the segment
            self.n_bad += 1
            if self.n_bad == 1:
                print(f"[capture] WARN: skipping bad image: {exc}", flush=True)
            return
        try:
            self.jobs.put_nowait((_ns(msg.header.stamp), img))
        except queue.Full:
            self.n_dropped += 1

    def _write_loop(self) -> None:
        while True:
            stamp, img = self.jobs.get()
            if img is None:
                return
            name = f"{stamp}.jpg"
            cv2.imwrite(os.path.join(self.dir, "frames", name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with self.lock:
                self.frames_csv.write(f"{stamp},frames/{name},{img.shape[1]},{img.shape[0]}\n")
                self.n_frames += 1
                self.size = (img.shape[1], img.shape[0])
            if self.a.segment == "checkerboard":
                self._board(img)

    def _board(self, img: np.ndarray) -> None:
        # Detect with the long side at ~640 px for speed (only feedback; the
        # offline tool re-detects at full size). Never upscale: a fixed 0.5
        # shrink made boards in small frames undetectable.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k = min(1.0, 640.0 / max(gray.shape))
        small = cv2.resize(gray, None, fx=k, fy=k) if k < 1.0 else gray
        ok, corners = cv2.findChessboardCorners(small, self.board, flags=cv2.CALIB_CB_FAST_CHECK)
        if not ok:
            return
        h, w = small.shape
        with self.lock:
            self.boards += 1
            for c in corners.reshape(-1, 2):
                self.grid[min(3, int(4 * c[1] / h)), min(3, int(4 * c[0] / w))] += 1

    def on_cloud(self, msg: PointCloud2) -> None:
        pts = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        xyz = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
        stamp = _ns(msg.header.stamp)
        name = f"{stamp}.npy"
        np.save(os.path.join(self.dir, "clouds", name), xyz)
        with self.lock:
            self.clouds_csv.write(f"{stamp},clouds/{name},{msg.header.frame_id},{len(xyz)}\n")
            self.n_clouds += 1
            self.last_points = len(xyz)

    def on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        wz = msg.twist.twist.angular.z
        with self.lock:
            self.odom_csv.write(f"{_ns(msg.header.stamp)},{p.x},{p.y},{p.z},{q.x},{q.y},{q.z},{q.w},{wz}\n")
            self.n_odom += 1
            self.last_wz = wz

    def coverage(self) -> float:
        return float(np.count_nonzero(self.grid)) / self.grid.size

    def status(self) -> str:
        with self.lock:
            s = (f"frames {self.n_frames} (dropped {self.n_dropped}, bad {self.n_bad}) clouds {self.n_clouds} "
                 f"({self.last_points} pts) odom {self.n_odom}")
            if self.a.segment == "checkerboard":
                rows = " | ".join("".join("#" if v else "." for v in r) for r in self.grid)
                s += f"  boards {self.boards}  coverage {self.coverage():.2f} [{rows}]"
            if self.a.segment == "yaw":
                s += f"  yaw rate {self.last_wz:+.2f} rad/s"
        return s

    def done_early(self) -> bool:
        return (self.a.segment == "checkerboard" and self.boards >= self.a.min_views
                and self.coverage() >= 0.75)

    def close(self, meta: dict) -> None:
        self.jobs.put((0, None))
        self.writer.join(timeout=30)
        for f in (self.frames_csv, self.clouds_csv, self.odom_csv):
            f.close()
        with open(os.path.join(self.dir, "meta.yaml"), "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segment", choices=["checkerboard", "scene", "yaw"], required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument("--latency-s", type=float, default=0.0, help="latency_s the camera node is running with")
    ap.add_argument("--image-rate", type=float, default=0.0, help="max saved frames per second (0 = all)")
    ap.add_argument("--board", default="7x6")
    ap.add_argument("--min-views", type=int, default=30)
    ap.add_argument("--taped", nargs="*", default=[], help="label:x:y[:z] in base_link, metres")
    ap.add_argument("--notes", default="")
    ap.add_argument("--image-topic", default="/camera/front/image_raw")
    ap.add_argument("--cloud-topic", default="/go2/lidar/points")
    ap.add_argument("--odom-topic", default="/odom")
    a = ap.parse_args()

    with open(a.intrinsics) as f:
        intr = yaml.safe_load(f)
    with open(a.extrinsics) as f:
        ext = yaml.safe_load(f)
    meta = {
        "segment": a.segment,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "camera_frame": ext.get("child_frame", "front_camera_optical_frame"),
        "latency_s": a.latency_s,
        "fixed_frame": "odom",
        "intrinsics": intr,
        "extrinsics": ext,
        "notes": a.notes,
    }
    if a.taped:
        meta["taped_objects"] = parse_taped(a.taped)

    rclpy.init()
    node = rclpy.create_node(f"capture_recorder_{a.segment}")
    rec = Recorder(node, a)
    t_end = time.monotonic() + a.seconds
    t_print = 0.0
    print(f"[capture] {a.segment}: recording up to {a.seconds:.0f} s into {rec.dir}", flush=True)
    try:
        while time.monotonic() < t_end and not rec.done_early():
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic() - t_print > 2.0:
                t_print = time.monotonic()
                print(f"[capture] {t_end - t_print:5.0f} s left  {rec.status()}", flush=True)
    except KeyboardInterrupt:
        pass
    meta["frame_size"] = list(rec.size) if rec.size else None
    rec.close(meta)
    node.destroy_node()
    rclpy.try_shutdown()
    print(f"[capture] {a.segment} done: {rec.status()}", flush=True)
    # A segment without both camera and LiDAR data is useless offline: say so now.
    if rec.n_frames == 0 or rec.n_clouds == 0 or rec.n_odom == 0:
        print("[capture] FAIL: missing frames, clouds or odometry", flush=True)
        return 2
    if a.segment == "checkerboard" and (rec.boards < 15 or rec.coverage() < 0.75):
        print("[capture] WARN: fewer than 15 boards or coverage < 0.75; intrinsics will likely fail", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
