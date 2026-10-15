"""Exact-stamp bridge from validated fusion positions to semantic mapping.

The join and conversion are independent of ROS so their rejection paths can be
checked offline. ROS imports are deferred until the optional executable starts.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np


class FusionPairError(ValueError):
    """A matched pair cannot safely supply mapper observations."""


def _stamp_ns(stamp):
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
        raise FusionPairError("Invalid image timestamp")
    value = stamp.sec * 1_000_000_000 + stamp.nanosec
    if value <= 0:
        raise FusionPairError("Image timestamp must be positive")
    return value


def _frame(frame):
    if not isinstance(frame, str) or not frame or len(frame) > 256:
        raise FusionPairError("Missing or excessive source frame")
    return frame


def semantic_key(message):
    frame = _frame(message.source_frame_id)
    if frame != message.header.frame_id:
        raise FusionPairError("Semantic header/source frames differ")
    return _stamp_ns(message.header.stamp), frame


def fusion_key(message):
    _frame(message.header.frame_id)
    return _stamp_ns(message.header.stamp), _frame(message.camera_frame)


def bridge_pair(semantic, fusion, *, map_frame="map", max_detections=256, max_embedding_dim=4096):
    """Copy semantic identity, replacing every admitted 3D position with fusion.

    Dimensions and depth become zero (unknown): FusionObject supplies neither.
    Original depth, extents and precomputed map centroids are never reused.
    """
    if semantic_key(semantic) != fusion_key(fusion):
        raise FusionPairError("Image stamps or camera frames differ")
    if fusion.status != 0:
        raise FusionPairError("Fusion batch is not valid")
    if len(semantic.detections) != len(fusion.objects):
        raise FusionPairError("Detection/fusion counts differ")
    if len(semantic.detections) > max_detections:
        raise FusionPairError("Detection capacity exceeded")
    accepted = []
    for detection, obj in zip(semantic.detections, fusion.objects):
        if not detection.label or len(detection.label) > 256 or detection.label != obj.label:
            raise FusionPairError("Detection/fusion labels differ")
        if not (
            math.isfinite(detection.score)
            and 0 <= detection.score <= 1
            and math.isfinite(obj.detector_score)
            and abs(detection.score - obj.detector_score) <= 1e-6
        ):
            raise FusionPairError("Detection/fusion scores differ or are invalid")
        if obj.status not in range(10) or bool(obj.valid) != (obj.status == 0):
            raise FusionPairError("Fusion object status/valid disagree")
        if not obj.valid:
            continue
        if obj.track_id <= 0 or obj.support_points <= 0:
            raise FusionPairError("Valid fusion object lacks track/support")
        position = [obj.position.x, obj.position.y, obj.position.z]
        if not all(math.isfinite(value) for value in position):
            raise FusionPairError("Nonfinite fused position")
        covariance = np.asarray(obj.covariance, dtype=np.float64)
        if covariance.size != 9 or not np.isfinite(covariance).all():
            raise FusionPairError("Invalid fusion covariance")
        covariance = covariance.reshape(3, 3)
        if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-10):
            raise FusionPairError("Asymmetric fusion covariance")
        if np.linalg.eigvalsh(covariance).min() < -1e-10:
            raise FusionPairError("Fusion covariance is not positive semidefinite")
        if not (
            0 < detection.embedding_dim <= max_embedding_dim
            and detection.embedding_dim == len(detection.embedding)
        ):
            raise FusionPairError("Invalid semantic embedding dimension")
        embedding = np.asarray(detection.embedding, dtype=np.float64)
        if not np.isfinite(embedding).all() or np.linalg.norm(embedding) <= 1e-12:
            raise FusionPairError("Invalid semantic embedding")
        accepted.append((detection, obj))

    result = deepcopy(semantic)
    result.header = deepcopy(fusion.header)
    result.source_frame_id = fusion.header.frame_id
    result.detections = []
    # The detector backprojection timing describes geometry we discarded.
    result.latency_backproject_ms = -1.0
    for detection, obj in accepted:
        converted = deepcopy(detection)
        converted.centroid_3d_camera = deepcopy(obj.position)
        converted.centroid_3d_map_valid = fusion.header.frame_id == map_frame
        converted.centroid_3d_map.x = obj.position.x if converted.centroid_3d_map_valid else 0.0
        converted.centroid_3d_map.y = obj.position.y if converted.centroid_3d_map_valid else 0.0
        converted.centroid_3d_map.z = obj.position.z if converted.centroid_3d_map_valid else 0.0
        converted.depth_median_m = 0.0
        converted.dimensions_xyz.x = 0.0
        converted.dimensions_xyz.y = 0.0
        converted.dimensions_xyz.z = 0.0
        result.detections.append(converted)
    return result


class FusionSemanticJoin:
    """Bounded, expiring join with per-camera replay suppression.

    Expiry uses monotonic receive time, independent of sensor and ROS clocks.
    Only the latest matched stamp per camera is retained, bounded by capacity.
    """

    def __init__(
        self,
        *,
        capacity=20,
        expiry_s=2.0,
        max_detections=256,
        max_embedding_dim=4096,
        map_frame="map",
        clock=time.monotonic,
    ):
        if not (1 <= capacity <= 1000 and math.isfinite(expiry_s) and 0 < expiry_s <= 60):
            raise ValueError("Invalid join capacity or expiry")
        if not (1 <= max_detections <= 256 and 1 <= max_embedding_dim <= 4096):
            raise ValueError("Invalid detection or embedding capacity")
        self.capacity = capacity
        self.expiry_s = expiry_s
        self.max_detections = max_detections
        self.max_embedding_dim = max_embedding_dim
        self.map_frame = _frame(map_frame)
        self.clock = clock
        self.semantic = OrderedDict()
        self.fusion = OrderedDict()
        self.last_matched = OrderedDict()

    def expire(self):
        now = self.clock()
        for cache in (self.semantic, self.fusion):
            for key, (received, _) in list(cache.items()):
                if now - received >= self.expiry_s or now < received:
                    del cache[key]

    def add_semantic(self, message):
        if len(message.detections) > self.max_detections:
            raise FusionPairError("Detection capacity exceeded")
        for detection in message.detections:
            if len(detection.embedding) > self.max_embedding_dim:
                raise FusionPairError("Embedding capacity exceeded")
        return self._add(semantic_key(message), message, self.semantic, self.fusion, True)

    def add_fusion(self, message):
        if len(message.objects) > self.max_detections:
            raise FusionPairError("Fusion capacity exceeded")
        return self._add(fusion_key(message), message, self.fusion, self.semantic, False)

    def _add(self, key, message, own, other, is_semantic):
        self.expire()
        stamp, camera = key
        if stamp <= self.last_matched.get(camera, 0):
            raise FusionPairError("Duplicate or out-of-order matched image")
        if key not in other:
            # A duplicate does not refresh its expiry or replace its payload.
            if key not in own:
                if len(own) >= self.capacity:
                    own.popitem(last=False)
                own[key] = self.clock(), message
            return None
        _, counterpart = other.pop(key)
        own.pop(key, None)
        self.last_matched[camera] = stamp
        self.last_matched.move_to_end(camera)
        if len(self.last_matched) > self.capacity:
            self.last_matched.popitem(last=False)
        semantic, fusion = (message, counterpart) if is_semantic else (counterpart, message)
        return bridge_pair(
            semantic,
            fusion,
            map_frame=self.map_frame,
            max_detections=self.max_detections,
            max_embedding_dim=self.max_embedding_dim,
        )


def make_adapter_node(*, parameter_overrides=None):
    """Create the optional ROS node, importing ROS only when requested."""
    from go2_fusion_cpp.msg import FusionArray
    from go2_semantic_msgs.msg import SemanticDetectionArray
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    class AdapterNode(Node):
        def __init__(self):
            super().__init__(
                "fusion_semantic_adapter", parameter_overrides=parameter_overrides or []
            )
            defaults = {
                "semantic_topic": "/semantic/detections",
                "fusion_topic": "/fusion/objects",
                "output_topic": "/semantic/fused_detections",
                "map_frame": "map",
                "cache_capacity": 20,
                "cache_expiry_s": 2.0,
                "max_detections": 256,
                "max_embedding_dim": 4096,
            }
            for name, default in defaults.items():
                self.declare_parameter(name, default)

            def param(name):
                return self.get_parameter(name).value

            topics = [
                self.resolve_topic_name(param(name))
                for name in ("semantic_topic", "fusion_topic", "output_topic")
            ]
            if len(set(topics)) != len(topics):
                raise ValueError("Adapter input/output topics must be distinct")
            self.join = FusionSemanticJoin(
                capacity=param("cache_capacity"),
                expiry_s=param("cache_expiry_s"),
                map_frame=param("map_frame"),
                max_detections=param("max_detections"),
                max_embedding_dim=param("max_embedding_dim"),
            )
            qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
            self.publisher = self.create_publisher(
                SemanticDetectionArray, param("output_topic"), qos
            )
            self.semantic_sub = self.create_subscription(
                SemanticDetectionArray, param("semantic_topic"), self.on_semantic, qos
            )
            self.fusion_sub = self.create_subscription(
                FusionArray, param("fusion_topic"), self.on_fusion, qos
            )
            self.expiry_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.timer = self.create_timer(
                min(0.25, self.join.expiry_s), self.join.expire, clock=self.expiry_clock
            )

        def receive(self, message, add):
            try:
                result = add(message)
            except FusionPairError as exc:
                self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
                return
            if result is not None:
                self.publisher.publish(result)

        def on_semantic(self, message):
            self.receive(message, self.join.add_semantic)

        def on_fusion(self, message):
            self.receive(message, self.join.add_fusion)

    return AdapterNode()


def main(args=None):
    """Start the optional adapter; it launches no sensor or motion nodes."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = make_adapter_node()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
