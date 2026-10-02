"""In-process ROS 2 tests for lidar_depth_node: real TF, real messages, no robot."""

import math
import time

import numpy as np
import pytest
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from go2_open_vocab_detector.qos_profiles import camera_image_qos, camera_info_qos
from go2_rgb_lidar.image_msg import fill_image, image_to_numpy
from go2_rgb_lidar.lidar_depth_node import LidarDepthNode
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

W, H, FX, CX, CY = 320, 240, 200.0, 160.0, 120.0
CAM_FRAME = "front_camera_optical_frame"


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def _write_calibration(tmp_path, nominal=False):
    intr = {
        "nominal": nominal,
        "image_width": W,
        "image_height": H,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [FX, 0, CX, 0, FX, CY, 0, 0, 1]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0, 0, 0, 0, 0]},
    }
    ext = {
        "nominal": nominal,
        "parent_frame": "base_link",
        "child_frame": CAM_FRAME,
        "xyz": [0.3, 0.0, 0.1],
        "rpy": [-math.pi / 2, 0.0, -math.pi / 2],
    }
    ip, ep = tmp_path / "intr.yaml", tmp_path / "ext.yaml"
    ip.write_text(yaml.safe_dump(intr))
    ep.write_text(yaml.safe_dump(ext))
    return str(ip), str(ep)


class Harness:
    """Drives the node: static odom->base_link (robot at (1, 2) facing +y), sinks for outputs."""

    def __init__(self, tmp_path, nominal=False, **params):
        ip, ep = _write_calibration(tmp_path, nominal)
        overrides = [
            Parameter("intrinsics_file", value=ip),
            Parameter("extrinsics_file", value=ep),
            Parameter("splat_radius_px", value=0),
            Parameter("occlusion_window_px", value=1),
            Parameter("max_rate_hz", value=0.0),
        ] + [Parameter(k, value=v) for k, v in params.items()]
        self.node = LidarDepthNode(parameter_overrides=overrides)
        self.io = rclpy.create_node("lidar_depth_test_io")
        self.tf = StaticTransformBroadcaster(self.io)
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = "odom", "base_link"
        t.transform.translation.x, t.transform.translation.y = 1.0, 2.0
        t.transform.rotation.z, t.transform.rotation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)
        self.tf.sendTransform(t)
        self.image_pub = self.io.create_publisher(Image, "/camera/front/image_raw", qos_profile_sensor_data)
        self.cloud_pub = self.io.create_publisher(PointCloud2, "/go2/lidar/points", qos_profile_sensor_data)
        self.depth, self.info, self.overlay = [], [], []
        self.io.create_subscription(Image, "/camera/front/lidar_depth", self.depth.append, camera_image_qos())
        # The detector's camera_info QoS (RELIABLE, depth 10): must match the publisher.
        self.io.create_subscription(CameraInfo, "/camera/front/camera_info", self.info.append, camera_info_qos())
        self.io.create_subscription(Image, "/camera/front/lidar_overlay", self.overlay.append, qos_profile_sensor_data)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self.ex.add_node(self.io)
        self.spin(0.3)  # discovery + static TF

    def spin(self, seconds, until=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.ex.spin_once(timeout_sec=0.02)
            if until is not None and until():
                return True
        return False

    def stamp(self, offset_s=0.0):
        return (self.io.get_clock().now() + rclpy.duration.Duration(nanoseconds=int(offset_s * 1e9))).to_msg()

    def send_cloud(self, points_odom, stamp):
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(Header(frame_id="odom", stamp=stamp), points_odom))

    def send_image(self, stamp, width=W, height=H, frame=CAM_FRAME):
        msg = fill_image(Image(), np.full((height, width, 3), 90, np.uint8), "bgr8")
        msg.header.stamp, msg.header.frame_id = stamp, frame
        self.image_pub.publish(msg)
        return msg

    def close(self):
        self.ex.shutdown()
        self.node.destroy_node()
        self.io.destroy_node()


def _odom_point(cam_x, cam_y, cam_z):
    """Camera-optical point -> odom, for the harness pose (camera at odom (1, 2.3, 0.1) facing +y)."""
    return [1.0 + cam_x, 2.3 + cam_z, 0.1 - cam_y]


@pytest.fixture
def harness(tmp_path, request):
    h = Harness(tmp_path, **getattr(request, "param", {}))
    yield h
    h.close()


def test_depth_and_info_are_aligned_to_the_colour_frame(harness):
    pts = [_odom_point(0.5, 0.0, 3.0), _odom_point(-0.6, 0.3, 2.0), _odom_point(0.0, 0.0, -2.0)]
    s = harness.stamp()
    harness.send_cloud(pts, s)
    harness.spin(0.2)
    img = harness.send_image(harness.stamp(0.05))
    assert harness.spin(2.0, until=lambda: harness.depth and harness.info)

    depth_msg, info = harness.depth[0], harness.info[0]
    assert depth_msg.header == img.header and info.header == img.header
    assert depth_msg.encoding == "16UC1" and (depth_msg.width, depth_msg.height) == (W, H)
    assert list(info.k) == [FX, 0, CX, 0, FX, CY, 0, 0, 1]
    depth = image_to_numpy(depth_msg)
    # (0.5, 0, 3) -> u = 200*0.5/3 + 160 = 193.3, v = 120
    assert depth[120, 193] == 3000
    # (-0.6, 0.3, 2) -> u = 100, v = 150
    assert depth[150, 100] == 2000
    assert np.count_nonzero(depth) == 2  # the point behind the camera is dropped


@pytest.mark.parametrize("harness", [{"nominal": True, "publish_overlay": True}], indirect=True)
def test_nominal_calibration_withholds_depth_but_keeps_overlay(harness):
    harness.send_cloud([_odom_point(0.0, 0.0, 3.0)], harness.stamp())
    harness.spin(0.2)
    harness.send_image(harness.stamp(0.05))
    assert harness.spin(2.0, until=lambda: harness.overlay)
    harness.spin(0.3)
    assert harness.depth == [] and harness.info == []
    assert harness.node.stats["dropped_nominal"] >= 1
    overlay = image_to_numpy(harness.overlay[0])
    assert overlay[120, 160].tolist() != [90, 90, 90]  # the LiDAR return is drawn


def test_stale_lidar_withholds_depth(harness):
    harness.send_cloud([_odom_point(0.0, 0.0, 3.0)], harness.stamp(-2.0))
    harness.spin(0.2)
    harness.send_image(harness.stamp())
    harness.spin(1.0)
    assert harness.depth == []
    assert harness.node.stats["dropped_no_cloud"] == 1


def test_no_lidar_at_all_withholds_depth(harness):
    harness.send_image(harness.stamp())
    harness.spin(0.5)
    assert harness.depth == []
    assert harness.node.stats["dropped_no_cloud"] == 1


def test_wrong_size_or_frame_withholds_depth(harness):
    harness.send_cloud([_odom_point(0.0, 0.0, 3.0)], harness.stamp())
    harness.spin(0.2)
    harness.send_image(harness.stamp(), width=640, height=480)
    harness.send_image(harness.stamp(), frame="camera_link")
    harness.spin(0.5)
    assert harness.depth == []
    assert harness.node.stats["dropped_size"] == 2


def test_extrinsic_is_live_tunable(harness):
    # Move the camera 0.5 m to the robot's left (base +y): the same point now
    # appears 0.5 m further right in the image.
    result = harness.node.set_parameters([Parameter("extrinsic_xyz", value=[0.3, 0.5, 0.1])])
    assert result[0].successful
    assert not harness.node.set_parameters([Parameter("extrinsic_rpy", value=[0.0, 1.0])])[0].successful
    harness.send_cloud([_odom_point(0.0, 0.0, 3.0)], harness.stamp())
    harness.spin(0.2)
    harness.send_image(harness.stamp(0.05))
    assert harness.spin(2.0, until=lambda: harness.depth)
    depth = image_to_numpy(harness.depth[0])
    ys, xs = np.nonzero(depth)
    # camera x = +0.5 at 3 m -> u = 200*0.5/3 + 160 = 193.3 -> 193
    assert (ys.tolist(), xs.tolist()) == ([120], [193])
