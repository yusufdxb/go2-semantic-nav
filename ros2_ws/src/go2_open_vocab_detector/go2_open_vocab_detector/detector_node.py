"""go2_open_vocab_detector — ROS 2 node that produces SemanticDetectionArray from RGB-D.

Subscribes to synchronized color + depth + camera_info. Runs the configured
detector → segmenter → encoder chain on each frame, back-projects masks to 3D
in the camera optical frame, and publishes a SemanticDetectionArray at
`detection_rate_hz` using a rate-limiting timer.

This node is intentionally "soft lifecycle" — it exposes configure/activate via
parameters (set the `active` param to true/false) rather than using the full
rclpy LifecycleNode machinery, matching the pattern used elsewhere in the GO2
stack.
"""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3
from go2_semantic_msgs.msg import SemanticDetection, SemanticDetectionArray
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from vision_msgs.msg import BoundingBox2D, Pose2D

from .backends import make_detector, make_encoder, make_segmenter
from .depth_backproject import (
    CameraIntrinsics,
    mask_rle_encode,
    object_centroid_3d,
)
from .qos_profiles import camera_image_qos, camera_info_qos, semantic_reliable_qos

_DEFAULT_PROMPTS = [
    "chair",
    "sofa",
    "table",
    "desk",
    "window",
    "door",
    "person",
    "refrigerator",
    "television",
    "bed",
    "lamp",
    "potted plant",
    "bookshelf",
    "cup",
    "laptop",
    "backpack",
]


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("go2_open_vocab_detector")

        # --- parameters ---
        self.declare_parameter("detector_backend", "yolo_world_v2_s")
        self.declare_parameter("segmenter_backend", "mobile_sam")
        self.declare_parameter("encoder_backend", "openclip_vit_b16")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("detection_rate_hz", 5.0)
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("max_objects_per_frame", 30)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("prompts", _DEFAULT_PROMPTS)
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("mask_downsample", 1)
        self.declare_parameter(
            "active",
            True,
            ParameterDescriptor(description="If false, callbacks short-circuit (lifecycle stand-in)."),
        )

        # --- state ---
        self._bridge = CvBridge()
        self._detector = None
        self._segmenter = None
        self._encoder = None
        self._intr: Optional[CameraIntrinsics] = None
        self._last_publish_ns: int = 0
        self._min_period_ns: int = 0
        self._prompts_cache: list[str] = []

        # --- pub/sub ---
        self._pub = self.create_publisher(
            SemanticDetectionArray, "/semantic/detections", semantic_reliable_qos()
        )

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        slop = float(self.get_parameter("sync_slop_sec").value)

        self._sub_color = message_filters.Subscriber(self, Image, image_topic, qos_profile=camera_image_qos())
        self._sub_depth = message_filters.Subscriber(self, Image, depth_topic, qos_profile=camera_image_qos())
        self._sub_info = message_filters.Subscriber(self, CameraInfo, info_topic, qos_profile=camera_info_qos())
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_color, self._sub_depth, self._sub_info], queue_size=5, slop=slop
        )
        self._sync.registerCallback(self._on_frame)

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._update_rate_limit()

        # --- lazy backend load ---
        try:
            self._load_backends()
        except Exception as exc:
            self.get_logger().error(f"Backend load failed; node will stay inactive: {exc}")
            self.get_logger().debug(traceback.format_exc())
            self.set_parameters([rclpy.parameter.Parameter("active", rclpy.Parameter.Type.BOOL, False)])

        self.get_logger().info(
            "go2_open_vocab_detector ready: "
            f"det={self.get_parameter('detector_backend').value}, "
            f"seg={self.get_parameter('segmenter_backend').value}, "
            f"enc={self.get_parameter('encoder_backend').value}, "
            f"device={self.get_parameter('device').value}, "
            f"rate={self.get_parameter('detection_rate_hz').value} Hz"
        )

    # --------------------------------------------------------------------- setup

    def _load_backends(self) -> None:
        det_name = str(self.get_parameter("detector_backend").value)
        seg_name = str(self.get_parameter("segmenter_backend").value)
        enc_name = str(self.get_parameter("encoder_backend").value)
        device = str(self.get_parameter("device").value)
        prompts = [str(p) for p in self.get_parameter("prompts").value]

        self._detector = make_detector(det_name)
        self._segmenter = make_segmenter(seg_name)
        self._encoder = make_encoder(enc_name)

        self._detector.load(device=device, prompts=prompts)
        self._segmenter.load(device=device)
        self._encoder.load(device=device)
        self._prompts_cache = list(prompts)

    def _update_rate_limit(self) -> None:
        rate = float(self.get_parameter("detection_rate_hz").value)
        self._min_period_ns = int(1e9 / max(rate, 0.1))

    def _on_set_parameters(self, params) -> SetParametersResult:
        names = {p.name for p in params}
        try:
            if "detection_rate_hz" in names:
                # Will be applied at next callback via _update_rate_limit
                pass
            if "prompts" in names and self._detector is not None:
                prompts = [str(p) for p in next(p for p in params if p.name == "prompts").value]
                self._detector.set_prompts(prompts)
                self._prompts_cache = prompts
            self._update_rate_limit()
            return SetParametersResult(successful=True)
        except Exception as exc:
            self.get_logger().error(f"Parameter update failed: {exc}")
            return SetParametersResult(successful=False, reason=str(exc))

    # ------------------------------------------------------------------- callback

    def _on_frame(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if not bool(self.get_parameter("active").value):
            return
        if self._detector is None or self._segmenter is None or self._encoder is None:
            return

        # Rate limit
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_publish_ns < self._min_period_ns:
            return

        try:
            color_bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            if depth_msg.encoding in {"16UC1", "mono16"}:
                depth_mm = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            else:
                # float depth in meters → convert to uint16 mm
                depth_m = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                depth_mm = (np.clip(depth_m, 0, 65.535) * 1000.0).astype(np.uint16)
        except Exception as exc:  # cv_bridge / encoding errors
            self.get_logger().warn(f"cv_bridge decode failed: {exc}")
            return

        if self._intr is None or self._intr.width != info_msg.width or self._intr.height != info_msg.height:
            self._intr = CameraIntrinsics.from_camera_info_k(
                list(info_msg.k), width=info_msg.width, height=info_msg.height
            )
            self.get_logger().info(
                f"camera intrinsics: fx={self._intr.fx:.1f} fy={self._intr.fy:.1f} "
                f"cx={self._intr.cx:.1f} cy={self._intr.cy:.1f} "
                f"{self._intr.width}x{self._intr.height}"
            )

        conf_threshold = float(self.get_parameter("conf_threshold").value)
        max_objects = int(self.get_parameter("max_objects_per_frame").value)
        min_depth_m = float(self.get_parameter("min_depth_m").value)
        max_depth_m = float(self.get_parameter("max_depth_m").value)

        # Inference chain ---------------------------------------------------
        t0_ns = time.perf_counter_ns()
        det_out = self._detector.detect(color_bgr, conf_threshold, max_objects)
        if det_out.boxes_xyxy.shape[0] == 0:
            self._publish_empty(color_msg.header, det_out, 0.0, 0.0, 0.0)
            self._last_publish_ns = now_ns
            return

        seg_out = self._segmenter.segment(color_bgr, det_out.boxes_xyxy)
        enc_out = self._encoder.encode_images(color_bgr, det_out.boxes_xyxy, seg_out.masks)

        t_bp0 = time.perf_counter_ns()
        # Align depth to color spatially: if depth and color have different sizes,
        # the RealSense driver typically publishes an aligned depth stream. If not,
        # scale masks to the depth image shape.
        depth_h, depth_w = depth_mm.shape
        color_h, color_w = color_bgr.shape[:2]
        if (depth_h, depth_w) != (color_h, color_w):
            depth_mm_r = cv2.resize(
                depth_mm, (color_w, color_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            depth_mm_r = depth_mm

        array_msg = SemanticDetectionArray()
        array_msg.header = color_msg.header
        array_msg.source_frame_id = color_msg.header.frame_id
        array_msg.detector_backend = str(self._detector.name)
        array_msg.segmenter_backend = str(self._segmenter.name)
        array_msg.encoder_backend = str(self._encoder.name)

        for i in range(det_out.boxes_xyxy.shape[0]):
            mask_full = seg_out.masks[i]

            centroid, depth_median, dims = object_centroid_3d(
                mask=mask_full,
                depth_image_mm=depth_mm_r,
                intr=self._intr,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
            if not np.isfinite(centroid).all():
                # Skip objects with unreliable depth
                continue

            # Tight ROI around mask for the RLE payload
            ys, xs = np.nonzero(mask_full)
            if ys.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            roi_mask = mask_full[y0:y1, x0:x1]

            det_msg = SemanticDetection()
            det_msg.label = det_out.labels[i]
            det_msg.score = float(det_out.scores[i])

            bbox = BoundingBox2D()
            cx = (det_out.boxes_xyxy[i, 0] + det_out.boxes_xyxy[i, 2]) / 2.0
            cy = (det_out.boxes_xyxy[i, 1] + det_out.boxes_xyxy[i, 3]) / 2.0
            w = det_out.boxes_xyxy[i, 2] - det_out.boxes_xyxy[i, 0]
            h = det_out.boxes_xyxy[i, 3] - det_out.boxes_xyxy[i, 1]
            pose = Pose2D()
            pose.position.x = float(cx)
            pose.position.y = float(cy)
            pose.theta = 0.0
            bbox.center = pose
            bbox.size_x = float(w)
            bbox.size_y = float(h)
            det_msg.bbox_2d = bbox

            roi = RegionOfInterest()
            roi.x_offset = x0
            roi.y_offset = y0
            roi.width = x1 - x0
            roi.height = y1 - y0
            roi.do_rectify = False
            det_msg.mask_roi = roi

            rle = mask_rle_encode(roi_mask.astype(bool))
            # ROS2 uint8[] supports values 0-255 directly via cast; counts are capped in chunks to 255.
            # We explicitly pack large counts into 255 chunks so each element fits in a uint8.
            packed: list[int] = []
            for count in rle:
                c = int(count)
                while c > 255:
                    packed.append(255)
                    packed.append(0)  # zero-length run of the opposite value
                    c -= 255
                packed.append(c)
            det_msg.mask_rle = packed

            det_msg.embedding = enc_out.image_embeddings[i].astype(np.float32).tolist()
            det_msg.embedding_dim = int(self._encoder.embedding_dim)

            det_msg.centroid_3d_camera = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]))
            det_msg.depth_median_m = float(depth_median)
            det_msg.dimensions_xyz = Vector3(x=float(dims[0]), y=float(dims[1]), z=float(dims[2]))

            det_msg.centroid_3d_map = Point()
            det_msg.centroid_3d_map_valid = False  # scene-graph node does TF
            det_msg.tracklet_id = f"det_{uuid.uuid4().hex[:8]}"

            array_msg.detections.append(det_msg)

        t_end_ns = time.perf_counter_ns()
        array_msg.latency_detector_ms = float(det_out.latency_ms)
        array_msg.latency_segmenter_ms = float(seg_out.latency_ms)
        array_msg.latency_encoder_ms = float(enc_out.latency_ms)
        array_msg.latency_backproject_ms = (t_end_ns - t_bp0) / 1e6
        array_msg.latency_total_ms = (t_end_ns - t0_ns) / 1e6

        self._pub.publish(array_msg)
        self._last_publish_ns = now_ns

    def _publish_empty(
        self,
        header,
        det_out,
        seg_latency_ms: float,
        enc_latency_ms: float,
        bp_latency_ms: float,
    ) -> None:
        msg = SemanticDetectionArray()
        msg.header = header
        msg.source_frame_id = header.frame_id
        if self._detector is not None:
            msg.detector_backend = str(self._detector.name)
        if self._segmenter is not None:
            msg.segmenter_backend = str(self._segmenter.name)
        if self._encoder is not None:
            msg.encoder_backend = str(self._encoder.name)
        msg.latency_detector_ms = float(det_out.latency_ms)
        msg.latency_segmenter_ms = float(seg_latency_ms)
        msg.latency_encoder_ms = float(enc_latency_ms)
        msg.latency_backproject_ms = float(bp_latency_ms)
        msg.latency_total_ms = float(det_out.latency_ms)
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
