"""capture_recorder over real ROS topics -> a capture the offline intrinsics tool accepts."""

import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "ros2_ws" / "src" / "go2_rgb_lidar" / "config"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def test_checkerboard_capture_feeds_offline_intrinsics(tmp_path, ros):
    boards = _load("board_views", ROOT / "scripts" / "calib" / "test_intrinsics_from_capture.py")
    itc = _load("itc", ROOT / "scripts" / "calib" / "intrinsics_from_capture.py")
    views = [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) if v.ndim == 2 else v for v in boards._views(40)]

    rec = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "capture_recorder.py"), "--out", str(tmp_path),
         "--segment", "checkerboard", "--seconds", "40", "--min-views", "25",
         "--intrinsics", str(CFG / "front_camera_intrinsics_nominal.yaml"),
         "--extrinsics", str(CFG / "front_camera_extrinsics_nominal.yaml")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    node = rclpy.create_node("capture_recorder_test_pub")
    img_pub = node.create_publisher(Image, "/camera/front/image_raw", qos_profile_sensor_data)
    cloud_pub = node.create_publisher(PointCloud2, "/go2/lidar/points", qos_profile_sensor_data)
    odom_pub = node.create_publisher(Odometry, "/odom", 50)
    cloud = np.array([[2.0, y, z] for y in np.linspace(-1, 1, 20) for z in np.linspace(-0.3, 0.5, 10)], np.float32)
    i = 0
    t0 = time.monotonic()
    while rec.poll() is None and time.monotonic() - t0 < 45:
        now = node.get_clock().now().to_msg()
        img = views[i % len(views)]
        msg = Image(header=Header(stamp=now, frame_id="front_camera_optical_frame"), height=img.shape[0],
                    width=img.shape[1], encoding="bgr8", step=img.shape[1] * 3, data=img.tobytes())
        img_pub.publish(msg)
        cloud_pub.publish(point_cloud2.create_cloud_xyz32(Header(stamp=now, frame_id="odom"), cloud))
        od = Odometry(header=Header(stamp=now, frame_id="odom"), child_frame_id="base_link")
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(0.05 * i), math.cos(0.05 * i)
        od.twist.twist.angular.z = 0.1
        odom_pub.publish(od)
        i += 1
        rclpy.spin_once(node, timeout_sec=0.1)
    out, _ = rec.communicate(timeout=30)
    node.destroy_node()
    assert rec.returncode == 0, out

    seg = tmp_path / "checkerboard"
    meta = yaml.safe_load((seg / "meta.yaml").read_text())
    assert meta["segment"] == "checkerboard" and meta["intrinsics"]["nominal"] is True
    assert "coverage" in out and "done" in out
    rows = (seg / "frames.csv").read_text().splitlines()
    assert rows[0] == "stamp_ns,file,width,height" and len(rows) > 25
    assert all((seg / r.split(",")[1]).exists() for r in rows[1:])
    clouds = (seg / "clouds.csv").read_text().splitlines()[1:]
    assert clouds and clouds[0].split(",")[2] == "odom" and np.load(seg / clouds[0].split(",")[1]).shape == (200, 3)
    odom = (seg / "odom.csv").read_text().splitlines()
    assert odom[0] == "stamp_ns,x,y,z,qx,qy,qz,qw,wz" and float(odom[1].split(",")[-1]) == pytest.approx(0.1)

    out_yaml = tmp_path / "intr.yaml"
    rc = itc.main(["--capture", str(tmp_path), "--board", "7x6", "--square", "0.025", "--out", str(out_yaml)])
    assert rc == 0 and out_yaml.exists()
    assert "nominal" not in yaml.safe_load(out_yaml.read_text())


def test_bad_image_is_skipped_not_fatal():
    rec = _load("capture_recorder", Path(__file__).parent / "capture_recorder.py")
    msg = Image(height=4, width=4, encoding="bgr8", step=12, data=bytes(10))
    with pytest.raises(ValueError):
        rec._bgr(msg)
    assert rec.parse_taped(["chair:1.5:0", "box:2:-0.5:0.3"]) == [
        {"label": "chair", "x_m": 1.5, "y_m": 0.0}, {"label": "box", "x_m": 2.0, "y_m": -0.5, "z_m": 0.3}]
