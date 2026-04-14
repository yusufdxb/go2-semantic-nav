"""QoS profiles matched to the GO2-seeing-eye-dog camera pipeline."""

from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def camera_image_qos() -> QoSProfile:
    """BEST_EFFORT + depth 1 — matches realsense2_camera driver and go2_perception."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def camera_info_qos() -> QoSProfile:
    """RELIABLE + depth 10 — matches go2_perception camera_info sub."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


def semantic_reliable_qos() -> QoSProfile:
    """RELIABLE + depth 5 — for our own semantic publications."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )
