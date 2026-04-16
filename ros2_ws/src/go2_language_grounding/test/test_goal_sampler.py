"""Unit tests for goal_sampler.sample_stand_off — costmap-aware ring sampling."""

from __future__ import annotations

import math

import numpy as np
from go2_language_grounding.goal_sampler import SamplerParams, sample_stand_off
from nav_msgs.msg import OccupancyGrid


def _empty_costmap(width: int = 200, height: int = 200, res: float = 0.05,
                   origin: tuple[float, float] = (-5.0, -5.0)) -> OccupancyGrid:
    cm = OccupancyGrid()
    cm.info.resolution = res
    cm.info.width = width
    cm.info.height = height
    cm.info.origin.position.x = origin[0]
    cm.info.origin.position.y = origin[1]
    cm.data = [0] * (width * height)
    return cm


def _occupied_costmap(width: int = 200, height: int = 200) -> OccupancyGrid:
    cm = _empty_costmap(width, height)
    cm.data = [100] * (width * height)
    return cm


def test_sampler_without_costmap_returns_ring_radius():
    target = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    ps = sample_stand_off(target, costmap=None, params=SamplerParams(stand_off_m=0.9, goal_ring_samples=8), map_frame="map")
    assert ps is not None
    dist = math.hypot(ps.pose.position.x - 1.0, ps.pose.position.y - 2.0)
    assert abs(dist - 0.9) < 1e-3


def test_sampler_with_free_costmap_accepts():
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cm = _empty_costmap()
    ps = sample_stand_off(target, costmap=cm, params=SamplerParams(stand_off_m=0.9, goal_ring_samples=8), map_frame="map")
    assert ps is not None
    assert ps.header.frame_id == "map"


def test_sampler_with_fully_occupied_costmap_rejects():
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cm = _occupied_costmap()
    ps = sample_stand_off(target, costmap=cm, params=SamplerParams(stand_off_m=0.9, goal_ring_samples=8, retry_with_wider_ring=False), map_frame="map")
    assert ps is None


def test_sampler_faces_target_from_sampled_pose():
    target = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    ps = sample_stand_off(target, costmap=None, params=SamplerParams(stand_off_m=1.0, goal_ring_samples=4), map_frame="map")
    assert ps is not None
    # Yaw should orient from (ps.x, ps.y) toward (target.x, target.y).
    dx = target[0] - ps.pose.position.x
    dy = target[1] - ps.pose.position.y
    expected_yaw = math.atan2(dy, dx)
    actual_yaw = 2 * math.atan2(ps.pose.orientation.z, ps.pose.orientation.w)
    # yaw from z,w quat components; normalize
    assert abs(math.atan2(math.sin(actual_yaw - expected_yaw), math.cos(actual_yaw - expected_yaw))) < 1e-3


def test_sampler_prefers_origin_closest_candidate():
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    origin = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    ps = sample_stand_off(
        target, costmap=None, params=SamplerParams(stand_off_m=1.0, goal_ring_samples=16),
        prefer_origin_xyz=origin, map_frame="map",
    )
    assert ps is not None
    # Should have picked the +x side (closer to origin) rather than far side.
    assert ps.pose.position.x > 0.0


def test_sampler_retry_with_wider_ring_rescues_tight_scene():
    """If the primary ring fails, a wider ring at 1.3× should rescue.

    We block a thick annulus around the primary ring (radius 0.9 ± 0.15 m) so both
    the 0.9 m samples and any close-to-ring cells are occupied, but the 1.17 m
    wider ring clears the annulus and succeeds.
    """
    cm = _empty_costmap()
    for i in range(cm.info.width):
        for j in range(cm.info.height):
            gx = cm.info.origin.position.x + i * cm.info.resolution
            gy = cm.info.origin.position.y + j * cm.info.resolution
            r = math.hypot(gx, gy)
            # Block annulus [0.7, 1.05] → primary ring 0.9 blocked, wider 1.17 free.
            if 0.70 <= r <= 1.05:
                cm.data[j * cm.info.width + i] = 100

    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p_no_retry = sample_stand_off(
        target, costmap=cm,
        params=SamplerParams(stand_off_m=0.9, goal_ring_samples=24, retry_with_wider_ring=False),
        map_frame="map",
    )
    p_retry = sample_stand_off(
        target, costmap=cm,
        params=SamplerParams(stand_off_m=0.9, goal_ring_samples=24, retry_with_wider_ring=True),
        map_frame="map",
    )
    assert p_no_retry is None, "primary ring should be blocked by the annulus"
    assert p_retry is not None, "wider ring should clear the annulus"
    r = math.hypot(p_retry.pose.position.x, p_retry.pose.position.y)
    assert r > 1.05, f"wider ring distance {r:.3f} should exceed the annulus outer edge 1.05"


def test_sampler_stand_off_override_clamped_to_range():
    params = SamplerParams(stand_off_m=0.9, goal_min_stand_off_m=0.4, goal_max_stand_off_m=1.5)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ps = sample_stand_off(target, costmap=None, params=params, stand_off_override=5.0, map_frame="map")
    assert ps is not None
    r = math.hypot(ps.pose.position.x, ps.pose.position.y)
    assert abs(r - 1.5) < 1e-3  # clamped to max
