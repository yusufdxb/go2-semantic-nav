"""lidar_depth_node: give the RGB-only GO2 front camera an aligned depth stream from the LiDAR.

    <image_topic>  (sensor_msgs/Image, camera optical frame, local clock)
    <cloud_topic>  (sensor_msgs/PointCloud2, any frame in TF, local clock)
        ->  <depth_topic>        16UC1 mm, 0 = no return, header == the colour frame's
        ->  <camera_info_topic>  from the calibration file, header == the colour frame's
        ->  static TF  <extrinsics parent> -> <camera optical frame>
        ->  <overlay_topic>      (optional) colour frame with LiDAR returns drawn on it

Depth and camera_info carry the colour frame's exact header, so the detector's
RGB-D synchroniser pairs them and its alignment gate passes. LiDAR clouds are
kept for ``accumulate_s`` in ``fixed_frame`` (odom) and transformed into the
camera at the image stamp, which densifies the sparse rings while the robot
moves.

Fail closed: no depth and no camera_info are published when the calibration is
nominal (unless ``allow_nominal_calibration``), when no cloud lies within
``max_cloud_gap_s`` of the image, when the image size differs from the
calibration, or when TF is unavailable. With no depth the detector produces no
objects and grounding refuses every query. The overlay is still published in
those cases, because it is the tool for fixing the calibration.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .calibration import CameraModel, Extrinsics, load_camera_model, load_extrinsics
from .image_msg import fill_image, image_to_numpy
from .projection import (
    CloudWindow,
    drop_occluded,
    invert_transform,
    make_transform,
    project_points,
    quaternion_to_matrix,
    render_depth_mm,
    rpy_to_matrix,
    transform_points,
)


def _transform_to_matrix(tf: TransformStamped) -> np.ndarray:
    q = tf.transform.rotation
    t = tf.transform.translation
    return make_transform(quaternion_to_matrix(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _quaternion_from_matrix(r: np.ndarray) -> tuple[float, float, float, float]:
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    return float(x), float(y), float(z), float(w)


class LidarDepthNode(Node):
    def __init__(self, *, parameter_overrides=None) -> None:
        super().__init__("go2_lidar_depth", parameter_overrides=parameter_overrides or [])
        p = self.declare_parameter
        p("image_topic", "/camera/front/image_raw")
        p("cloud_topic", "/go2/lidar/points")
        p("depth_topic", "/camera/front/lidar_depth")
        p("camera_info_topic", "/camera/front/camera_info")
        p("overlay_topic", "/camera/front/lidar_overlay")
        p("intrinsics_file", "")
        p("extrinsics_file", "")
        p("allow_nominal_calibration", False)
        p("fixed_frame", "odom")
        p("accumulate_s", 0.5)
        p("max_cloud_gap_s", 0.3)
        p("cloud_future_tolerance_s", 0.1)
        p("tf_wait_s", 0.15)
        p("max_rate_hz", 5.0)
        p("min_depth_m", 0.2)
        p("max_depth_m", 8.0)
        p("max_normalized_radius", 1.3)
        p("splat_radius_px", 2)
        p("occlusion_window_px", 21)
        p("occlusion_margin_m", 0.3)
        p("publish_overlay", False)
        # Live-tunable extrinsic overrides; empty = use the file. Lets the
        # overlay be lined up in the lab with `ros2 param set`.
        p("extrinsic_xyz", [0.0])
        p("extrinsic_rpy", [0.0])

        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self._cam: CameraModel = load_camera_model(str(g("intrinsics_file")))
        self._ext: Extrinsics = load_extrinsics(str(g("extrinsics_file")))
        self._nominal = self._cam.nominal or self._ext.nominal
        self._xyz = list(self._ext.xyz)
        self._rpy = list(self._ext.rpy)
        if len(g("extrinsic_xyz")) == 3:
            self._xyz = [float(v) for v in g("extrinsic_xyz")]
        if len(g("extrinsic_rpy")) == 3:
            self._rpy = [float(v) for v in g("extrinsic_rpy")]
        self._t_cam_parent = self._camera_from_parent()

        self._fixed = str(g("fixed_frame"))
        self._window = CloudWindow(float(g("accumulate_s")) + float(g("max_cloud_gap_s")) + 0.5)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._static = StaticTransformBroadcaster(self)
        self._broadcast_extrinsic()

        self._depth_pub = self.create_publisher(Image, str(g("depth_topic")), qos_profile_sensor_data)
        # RELIABLE: the detector subscribes to camera_info RELIABLE, and a
        # best-effort publisher never matches a reliable subscriber.
        self._info_pub = self.create_publisher(
            CameraInfo, str(g("camera_info_topic")),
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10),
        )
        self._overlay_pub = self.create_publisher(Image, str(g("overlay_topic")), qos_profile_sensor_data)
        self.create_subscription(Image, str(g("image_topic")), self._on_image, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, str(g("cloud_topic")), self._on_cloud, qos_profile_sensor_data)

        self._pending: Optional[Image] = None
        self._pending_since = 0.0
        self._last_accept = 0.0
        self.stats = {"published": 0, "dropped_tf": 0, "dropped_no_cloud": 0, "dropped_size": 0, "dropped_nominal": 0}
        self.last_valid_pixels = 0
        self.create_timer(0.01, self._process_pending)
        self.add_on_set_parameters_callback(self._on_params)

        if self._nominal and not bool(g("allow_nominal_calibration")):
            self.get_logger().error(
                "Front camera calibration is NOMINAL (not measured): depth and camera_info are "
                "withheld, so the detector will report nothing. Calibrate "
                "(docs/rgb_lidar_calibration.md) or set allow_nominal_calibration:=true for bench tests."
            )
        self.get_logger().info(
            f"lidar_depth ready: {self._cam.width}x{self._cam.height} fx={self._cam.fx:.1f}, "
            f"{self._ext.parent_frame}->{self._ext.child_frame} xyz={self._xyz} rpy={self._rpy}, "
            f"fixed_frame={self._fixed}, nominal={self._nominal}"
        )

    # ------------------------------------------------------------------ extrinsic
    def _camera_from_parent(self) -> np.ndarray:
        parent_from_camera = make_transform(rpy_to_matrix(*self._rpy), self._xyz)
        return invert_transform(parent_from_camera)

    def _broadcast_extrinsic(self) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._ext.parent_frame
        tf.child_frame_id = self._ext.child_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = self._xyz
        x, y, z, w = _quaternion_from_matrix(rpy_to_matrix(*self._rpy))
        tf.transform.rotation.x, tf.transform.rotation.y = x, y
        tf.transform.rotation.z, tf.transform.rotation.w = z, w
        self._static.sendTransform(tf)

    def _on_params(self, params) -> SetParametersResult:
        for prm in params:
            if prm.name in ("extrinsic_xyz", "extrinsic_rpy"):
                vals = list(prm.value)
                if len(vals) != 3:
                    return SetParametersResult(successful=False, reason=f"{prm.name} needs 3 values")
                if prm.name == "extrinsic_xyz":
                    self._xyz = [float(v) for v in vals]
                else:
                    self._rpy = [float(v) for v in vals]
        self._t_cam_parent = self._camera_from_parent()
        self._broadcast_extrinsic()
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ inputs
    def _on_cloud(self, msg: PointCloud2) -> None:
        pts = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        xyz = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float64)
        if msg.header.frame_id != self._fixed:
            try:
                tf = self._tf_buffer.lookup_transform(self._fixed, msg.header.frame_id, Time.from_msg(msg.header.stamp))
            except TransformException as exc:
                self.get_logger().warn(f"cloud dropped, no TF {msg.header.frame_id}->{self._fixed}: {exc}",
                                       throttle_duration_sec=5.0)
                return
            xyz = transform_points(_transform_to_matrix(tf), xyz)
        self._window.add(_stamp_ns(msg.header.stamp), xyz.astype(np.float32))

    def _on_image(self, msg: Image) -> None:
        now = time.monotonic()
        max_rate = float(self.get_parameter("max_rate_hz").value)
        if max_rate > 0 and now - self._last_accept < 1.0 / max_rate:
            return
        self._last_accept = now
        self._pending = msg
        self._pending_since = now
        self._process_pending()

    # ------------------------------------------------------------------ core
    def _process_pending(self) -> None:
        msg = self._pending
        if msg is None:
            return
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        stamp_ns = _stamp_ns(msg.header.stamp)

        if (msg.width, msg.height) != (self._cam.width, self._cam.height):
            self._pending = None
            self.stats["dropped_size"] += 1
            self.get_logger().error(
                f"image {msg.width}x{msg.height} does not match calibration "
                f"{self._cam.width}x{self._cam.height}; no depth published", throttle_duration_sec=5.0)
            return
        if msg.header.frame_id != self._ext.child_frame:
            self._pending = None
            self.stats["dropped_size"] += 1
            self.get_logger().error(
                f"image frame {msg.header.frame_id!r} is not the calibrated camera frame "
                f"{self._ext.child_frame!r}; no depth published", throttle_duration_sec=5.0)
            return

        try:
            tf = self._tf_buffer.lookup_transform(self._ext.parent_frame, self._fixed, Time.from_msg(msg.header.stamp))
        except TransformException as exc:
            if time.monotonic() - self._pending_since < float(g("tf_wait_s")):
                return  # odometry for this stamp may still be in flight
            self._pending = None
            self.stats["dropped_tf"] += 1
            self.get_logger().warn(f"no TF {self._fixed}->{self._ext.parent_frame} at image stamp: {exc}",
                                   throttle_duration_sec=5.0)
            return

        gap = self._window.nearest_gap_s(stamp_ns)
        if gap > float(g("max_cloud_gap_s")):
            if gap != float("inf") and time.monotonic() - self._pending_since < float(g("tf_wait_s")):
                return  # the cloud for this stamp may still be in flight
            self._pending = None
            self.stats["dropped_no_cloud"] += 1
            self.get_logger().warn(f"no LiDAR cloud within {g('max_cloud_gap_s')} s of the image "
                                   f"(nearest {gap:.2f} s); no depth published", throttle_duration_sec=5.0)
            return
        self._pending = None

        start = stamp_ns - int(float(g("accumulate_s")) * 1e9)
        end = stamp_ns + int(float(g("cloud_future_tolerance_s")) * 1e9)
        pts_fixed = self._window.points_between(start, end)
        t_cam_fixed = self._t_cam_parent @ _transform_to_matrix(tf)
        pts_cam = transform_points(t_cam_fixed, pts_fixed.astype(np.float64))
        u, v, z = project_points(pts_cam, self._cam, float(g("min_depth_m")), float(g("max_depth_m")),
                                 float(g("max_normalized_radius")))
        depth = render_depth_mm(u, v, z, self._cam.width, self._cam.height, int(g("splat_radius_px")))
        depth = drop_occluded(depth, int(g("occlusion_window_px")), float(g("occlusion_margin_m")))
        self.last_valid_pixels = int(np.count_nonzero(depth))

        if bool(g("publish_overlay")):
            self._publish_overlay(msg, depth)

        if self._nominal and not bool(g("allow_nominal_calibration")):
            self.stats["dropped_nominal"] += 1
            return
        depth_msg = fill_image(Image(), depth, "16UC1")
        depth_msg.header = msg.header
        self._depth_pub.publish(depth_msg)
        self._info_pub.publish(self._camera_info(msg))
        self.stats["published"] += 1

    def _camera_info(self, image: Image) -> CameraInfo:
        info = CameraInfo()
        info.header = image.header
        info.width, info.height = self._cam.width, self._cam.height
        info.distortion_model = self._cam.distortion_model
        info.d = [float(v) for v in self._cam.d]
        info.k = [float(v) for v in self._cam.k.ravel()]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(v) for v in self._cam.p.ravel()]
        return info

    def _publish_overlay(self, image: Image, depth_mm: np.ndarray) -> None:
        try:
            bgr = image_to_numpy(image)
        except ValueError as exc:
            self.get_logger().warn(f"overlay skipped: {exc}", throttle_duration_sec=5.0)
            return
        if bgr.ndim == 2:
            bgr = np.repeat(bgr[:, :, None], 3, axis=2)
        hit = depth_mm > 0
        # Near = red, far = blue, linear over [min_depth, max_depth].
        lo = float(self.get_parameter("min_depth_m").value) * 1000.0
        hi = float(self.get_parameter("max_depth_m").value) * 1000.0
        t = np.clip((depth_mm[hit].astype(np.float32) - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        bgr = bgr.copy()
        bgr[hit] = np.column_stack([255 * t, 64 * np.ones_like(t), 255 * (1 - t)]).astype(np.uint8)
        out = fill_image(Image(), bgr, "bgr8")
        out.header = image.header
        self._overlay_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarDepthNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
