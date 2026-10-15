#!/usr/bin/env python3
"""Send-to-subscriber latency of a front-camera driver, measured from the pixels.

Starts the counter sender and the camera node under test as separate
processes, subscribes to the image in this process (rclpy only: no GStreamer
here), reads each frame's counter from its pixels and reports, per frame:

  e2e      = subscriber receive time - sender hand-off time (what a consumer sees)
  stamp    = header.stamp - sender hand-off time (timestamp error vs capture)

plus received fps, frames lost, and the node's CPU use. Everything runs on one
machine on the loopback interface.

    python3 scripts/bench/bench_front_camera.py --arm cpp_avdec --seconds 30
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(__file__))
from rtp_counter_sender import BITS, BLOCK, PITCH, ROWS, X0  # noqa: E402

PY_DEFAULT = ("udpsrc address=127.0.0.1 port={port} caps=\"application/x-rtp, media=video, clock-rate=90000, "
              "encoding-name=H264\" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR "
              "! appsink name=sink drop=true max-buffers=1 sync=false")
ARMS = {
    # The Python node as committed (default libav threading).
    "py_avdec": ["ros2", "run", "go2_rgb_lidar", "front_camera_node", "--ros-args",
                 "-p", "address:=127.0.0.1", "-p", "port:={port}", "-p", "max_rate_hz:=0.0"],
    # The C++ node with its low-latency software decode.
    "cpp_avdec": ["ros2", "run", "go2_front_camera_cpp", "front_camera_node", "--ros-args",
                  "-p", "address:=127.0.0.1", "-p", "port:={port}", "-p", "decoder:=avdec"],
    # The C++ node with the Python node's decoder settings: separates language from decoder.
    "cpp_avdec_default": ["ros2", "run", "go2_front_camera_cpp", "front_camera_node", "--ros-args",
                          "-p", "pipeline:=" + PY_DEFAULT],
    # Smaller published image: fewer DDS fragments per frame.
    "cpp_avdec_540": ["ros2", "run", "go2_front_camera_cpp", "front_camera_node", "--ros-args",
                      "-p", "address:=127.0.0.1", "-p", "port:={port}", "-p", "decoder:=avdec",
                      "-p", "output_width:=960", "-p", "output_height:=540"],
    # Jetson only: hardware decoder.
    "cpp_nvv4l2": ["ros2", "run", "go2_front_camera_cpp", "front_camera_node", "--ros-args",
                   "-p", "address:=127.0.0.1", "-p", "port:={port}", "-p", "decoder:=nvv4l2"],
}


def read_counter(img: np.ndarray, sender_width: int = 1280):
    """Counter drawn by the sender at sender_width; scales if the node resized the image."""
    k = img.shape[1] / sender_width
    r = max(2, int(8 * k))
    vals = []
    for y0 in ROWS:
        v = 0
        cy = int((y0 + BLOCK // 2) * k)
        for i in range(BITS):
            cx = int((X0 + i * PITCH + BLOCK // 2) * k)
            if img[cy - r:cy + r, cx - r:cx + r].mean() > 128:
                v |= 1 << i
        vals.append(v)
    return vals[0] if vals[0] == vals[1] else None


def _running(*programs: str) -> list:
    """PIDs whose program (argv[0], or the script for an interpreter) ends with one of `programs`.

    Matches the executable, not the whole command line: `pgrep -f` also
    matches any shell whose command text merely mentions the name.
    """
    found = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        heads = [a.decode(errors="replace") for a in argv[:2]]
        if any(h.endswith(p) for h in heads for p in programs):
            found.append(int(d))
    return found


def _descendants(pid: int) -> list:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += [int(c) for c in subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()]
    return out


def cpu_seconds(pid: int) -> float:
    total = 0.0
    tick = os.sysconf("SC_CLK_TCK")
    for p in _descendants(pid):
        try:
            f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / tick
        except (FileNotFoundError, IndexError):
            pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=56200)
    ap.add_argument("--out", default="")
    ap.add_argument("--noise", action="store_true", help="incompressible sender frames")
    a = ap.parse_args()

    stray = _running("front_camera_node", "rtp_counter_sender.py")
    if stray:
        # Several receivers on one UDP port split the packets between them,
        # which shows up as frame loss in every arm.
        print(f"refusing: camera/sender processes already running: {stray}", file=sys.stderr)
        return 3
    tmp = tempfile.mkdtemp(prefix="bench_cam_")
    send_log = os.path.join(tmp, "send.csv")
    cmd = [c.replace("{port}", str(a.port)) for c in ARMS[a.arm]]
    # Own process group: `ros2 run` does not forward SIGTERM to the node it starts.
    node = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(2.0)
    rclpy.init()
    sub_node = rclpy.create_node("bench_front_camera")
    recv = []  # (recv_ns, stamp_ns, counter)

    def on_image(msg: Image) -> None:
        t = time.time_ns()
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        recv.append((t, msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, read_counter(img)))

    sub_node.create_subscription(Image, "/camera/front/image_raw", on_image, qos_profile_sensor_data)
    node_stats = []
    sub_node.create_subscription(String, "/camera/front/latency", lambda m: node_stats.append(json.loads(m.data)), 10)
    sender = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "rtp_counter_sender.py"),
                               "--port", str(a.port), "--seconds", str(a.seconds + a.warmup), "--log", send_log]
                              + (["--noise"] if a.noise else []))
    t_end = time.monotonic() + a.seconds + a.warmup + 1.5
    cpu0, t_cpu0 = None, None
    while time.monotonic() < t_end:
        rclpy.spin_once(sub_node, timeout_sec=0.01)
        if cpu0 is None and time.monotonic() > t_end - a.seconds - 1.5:
            cpu0, t_cpu0 = cpu_seconds(node.pid), time.monotonic()
    cpu = (cpu_seconds(node.pid) - cpu0) / (time.monotonic() - t_cpu0) * 100.0
    sender.wait(timeout=10)
    os.killpg(node.pid, signal.SIGINT)
    try:
        node.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(node.pid, signal.SIGKILL)
        node.wait(timeout=5)
    time.sleep(0.5)
    left = _running("front_camera_node")
    if left:
        print(f"warning: camera node still running after the run: {left}", file=sys.stderr)
    sub_node.destroy_node()
    rclpy.shutdown()

    sent = {int(r["counter"]): int(r["send_ns"]) for r in csv.DictReader(open(send_log))}
    warm = int(a.warmup * 15)
    e2e, stamp_err, bad = [], [], 0
    seen = set()
    for t, stamp, c in recv:
        if c is None or c not in sent:
            bad += 1
            continue
        if c < warm:
            continue
        seen.add(c)
        e2e.append((t - sent[c]) / 1e6)
        stamp_err.append((stamp - sent[c]) / 1e6)
    expected = [c for c in sent if c >= warm]
    pct = lambda v, q: float(np.percentile(v, q)) if v else float("nan")  # noqa: E731
    res = {
        "arm": a.arm,
        "content": "noise" if a.noise else "camera-like",
        "frames_expected": len(expected),
        "frames_received": len(seen),
        "lost_pct": round(100.0 * (1 - len(seen) / max(1, len(expected))), 2),
        "unreadable": bad,
        "e2e_ms": {"p50": round(pct(e2e, 50), 2), "p95": round(pct(e2e, 95), 2), "max": round(max(e2e, default=float("nan")), 2)},
        "stamp_minus_send_ms": {"p50": round(pct(stamp_err, 50), 2), "p95": round(pct(stamp_err, 95), 2)},
        "node_cpu_pct": round(cpu, 1),
        "rmw": os.environ.get("RMW_IMPLEMENTATION", "default"),
    }
    if node_stats:  # C++ node only: frames it published, vs what reached the subscriber
        res["node_published_total"] = node_stats[-1]["frames"]
        res["subscriber_received_total"] = len(recv)
        res["node_arrival_to_publish_ms_p95_last"] = node_stats[-1].get("arrival_to_publish_ms", {}).get("p95")
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    return 0 if e2e else 2


if __name__ == "__main__":
    sys.exit(main())
