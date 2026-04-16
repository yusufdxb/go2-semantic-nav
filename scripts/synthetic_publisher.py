#!/usr/bin/env python3
"""Publish a synthetic RGB-D stream plus TF for hardware-free integration tests.

Topics published:
  /camera/color/image_raw        sensor_msgs/Image (bgr8)
  /camera/depth/image_rect_raw   sensor_msgs/Image (16UC1, mm)
  /camera/color/camera_info      sensor_msgs/CameraInfo

TF broadcast:
  map -> odom -> base_link -> camera_color_optical_frame   (all static for this smoke test)

Source image: by default, ultralytics' shipped `bus.jpg`, which has people and a bus
that YOLO-World v2-s will reliably detect. A constant depth of `depth_m` meters is
overlaid so the back-projection has something to work with.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster


def _best_effort_qos() -> QoSProfile:
    q = QoSProfile(depth=1)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    return q


def _reliable_qos() -> QoSProfile:
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.RELIABLE
    q.history = HistoryPolicy.KEEP_LAST
    return q


class SyntheticPublisher(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_rgbd_publisher")

        self.declare_parameter("image_path", "")
        self.declare_parameter("image_dir", "",
                               ignore_override=False)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("depth_m", 2.0)
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("map_frame", "map")

        image_path = str(self.get_parameter("image_path").value).strip()
        image_dir = str(self.get_parameter("image_dir").value).strip()

        self._img_cycle: list[np.ndarray] = []
        if image_dir:
            import glob
            from pathlib import Path as _P

            paths = sorted(glob.glob(str(_P(image_dir) / "*.jpg"))
                           + glob.glob(str(_P(image_dir) / "*.png"))
                           + glob.glob(str(_P(image_dir) / "*.jpeg")))
            if not paths:
                raise FileNotFoundError(f"no images found in {image_dir}")
            self._img_cycle = [self._load_image(p) for p in paths]
            self.get_logger().info(f"cycling {len(self._img_cycle)} images from {image_dir}")
        else:
            if not image_path:
                image_path = self._find_sample_image()
            self._img_cycle = [self._load_image(image_path)]
        self._img = self._img_cycle[0]
        self._cycle_idx = 0
        self._bridge = CvBridge()

        self._pub_color = self.create_publisher(Image, "/camera/color/image_raw", _best_effort_qos())
        self._pub_depth = self.create_publisher(Image, "/camera/depth/image_rect_raw", _best_effort_qos())
        self._pub_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", _reliable_qos())

        self._tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        rate = float(self.get_parameter("rate_hz").value)
        self._timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info(
            f"synthetic_rgbd_publisher: img={image_path} shape={self._img.shape} "
            f"rate={rate} Hz depth={self.get_parameter('depth_m').value} m"
        )

    def _find_sample_image(self) -> str:
        """Locate a real indoor image via ultralytics' asset bundle or a fallback."""
        candidates = [
            Path.home() / ".config/Ultralytics/assets/bus.jpg",
            Path("/usr/lib/python3/dist-packages/ultralytics/assets/bus.jpg"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # Try the pip-installed ultralytics package
        try:
            import ultralytics
            assets = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
            if assets.exists():
                return str(assets)
        except Exception:
            pass
        raise FileNotFoundError(
            "no sample image found; pass --image_path or put one at ~/.config/Ultralytics/assets/bus.jpg"
        )

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        # Resize to RealSense native 640x480 for a realistic shape.
        return cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    def _publish_static_tf(self) -> None:
        map_frame = str(self.get_parameter("map_frame").value)
        camera_frame = str(self.get_parameter("camera_frame").value)
        stamps = self.get_clock().now().to_msg()

        # map -> odom (identity)
        t1 = TransformStamped()
        t1.header.stamp = stamps
        t1.header.frame_id = map_frame
        t1.child_frame_id = "odom"
        t1.transform.rotation.w = 1.0

        # odom -> base_link (identity at origin)
        t2 = TransformStamped()
        t2.header.stamp = stamps
        t2.header.frame_id = "odom"
        t2.child_frame_id = "base_link"
        t2.transform.rotation.w = 1.0

        # base_link -> camera_color_optical_frame
        # Standard optical-frame convention: z-forward, x-right, y-down.
        # Base-link is x-forward, y-left, z-up. Rotation q = (x=-0.5, y=0.5, z=-0.5, w=0.5)
        # puts the camera facing forward.
        t3 = TransformStamped()
        t3.header.stamp = stamps
        t3.header.frame_id = "base_link"
        t3.child_frame_id = camera_frame
        t3.transform.translation.x = 0.15  # 15 cm forward of base
        t3.transform.translation.z = 0.20  # 20 cm up
        t3.transform.rotation.x = -0.5
        t3.transform.rotation.y = 0.5
        t3.transform.rotation.z = -0.5
        t3.transform.rotation.w = 0.5

        self._tf.sendTransform([t1, t2, t3])

    def _tick(self) -> None:
        # Rotate through the cycle once per tick when more than one image is loaded.
        if len(self._img_cycle) > 1:
            self._cycle_idx = (self._cycle_idx + 1) % len(self._img_cycle)
            self._img = self._img_cycle[self._cycle_idx]

        now = self.get_clock().now().to_msg()
        color_msg = self._bridge.cv2_to_imgmsg(self._img, encoding="bgr8")
        color_msg.header.stamp = now
        color_msg.header.frame_id = str(self.get_parameter("camera_frame").value)
        self._pub_color.publish(color_msg)

        depth_m = float(self.get_parameter("depth_m").value)
        depth = np.full((self._img.shape[0], self._img.shape[1]), int(depth_m * 1000), dtype=np.uint16)
        # Put a mild gradient so centroids vary across image regions.
        for y in range(depth.shape[0]):
            depth[y, :] += int(200 * math.sin(y * math.pi / depth.shape[0]))
        depth_msg = self._bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = str(self.get_parameter("camera_frame").value)
        self._pub_depth.publish(depth_msg)

        info = CameraInfo()
        info.header.stamp = now
        info.header.frame_id = str(self.get_parameter("camera_frame").value)
        info.width = self._img.shape[1]
        info.height = self._img.shape[0]
        # Plausible RealSense D435i intrinsics at 640x480.
        info.k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
        info.p = [600.0, 0.0, 320.0, 0.0, 0.0, 600.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.distortion_model = "plumb_bob"
        self._pub_info.publish(info)


def main(args=None) -> int:
    # Handle SIGTERM cleanly (set by parent shells / `timeout`).
    import signal

    def _on_term(_signum, _frame):
        try:
            rclpy.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _on_term)

    try:
        rclpy.init(args=args)
    except RuntimeError:
        # Already initialized (rare; double-invoke guard).
        pass

    node = SyntheticPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception as exc:
        # Don't noisy-trace on shutdown races.
        if "context" not in str(exc).lower():
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
