"""Sample a reachable stand-off pose around a target object, optionally gated by a costmap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid


@dataclass
class SamplerParams:
    stand_off_m: float = 0.9
    goal_ring_samples: int = 12
    goal_min_stand_off_m: float = 0.4
    goal_max_stand_off_m: float = 2.0
    costmap_free_max_cost: int = 50
    retry_with_wider_ring: bool = True


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def _world_to_grid(
    x: float, y: float, costmap: OccupancyGrid
) -> Optional[tuple[int, int]]:
    info = costmap.info
    if info.resolution <= 0.0 or info.width == 0 or info.height == 0:
        return None
    gx = int((x - info.origin.position.x) / info.resolution)
    gy = int((y - info.origin.position.y) / info.resolution)
    if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
        return None
    return gx, gy


def _cell_cost(costmap: OccupancyGrid, gx: int, gy: int) -> int:
    idx = gy * costmap.info.width + gx
    if idx < 0 or idx >= len(costmap.data):
        return 100
    return int(costmap.data[idx])


def _is_free(costmap: Optional[OccupancyGrid], x: float, y: float, max_cost: int) -> bool:
    if costmap is None:
        return True  # no costmap → don't block; caller is responsible
    pt = _world_to_grid(x, y, costmap)
    if pt is None:
        return False  # off-map
    cost = _cell_cost(costmap, *pt)
    if cost < 0:  # unknown
        return False
    return cost <= max_cost


def sample_stand_off(
    target_xyz: np.ndarray,
    costmap: Optional[OccupancyGrid],
    params: SamplerParams | None = None,
    prefer_origin_xyz: Optional[np.ndarray] = None,
    stand_off_override: Optional[float] = None,
    map_frame: str = "map",
) -> Optional[PoseStamped]:
    """Return a PoseStamped standing `stand_off_m` from the target, facing it,
    landing on a free-ish costmap cell. Returns None if no sample passes.

    ``prefer_origin_xyz`` is the robot's current pose (if known); sampled poses
    are ranked by closeness to it so the robot picks the nearest reachable slot.
    """
    p = params or SamplerParams()
    radius = stand_off_override or p.stand_off_m
    radius = max(p.goal_min_stand_off_m, min(p.goal_max_stand_off_m, radius))

    candidates: list[tuple[float, PoseStamped]] = []

    # Two sweeps: primary ring at requested radius, secondary at 1.3× if needed.
    rings = [radius]
    if p.retry_with_wider_ring:
        rings.append(min(radius * 1.3, p.goal_max_stand_off_m))
    # Deduplicate if both rings landed on same radius.
    rings = list(dict.fromkeys(round(r, 3) for r in rings))

    for ring_radius in rings:
        for k in range(p.goal_ring_samples):
            theta = 2.0 * math.pi * k / p.goal_ring_samples
            gx = float(target_xyz[0] + ring_radius * math.cos(theta))
            gy = float(target_xyz[1] + ring_radius * math.sin(theta))
            if not _is_free(costmap, gx, gy, p.costmap_free_max_cost):
                continue
            # Face the target from this pose.
            yaw = math.atan2(float(target_xyz[1]) - gy, float(target_xyz[0]) - gx)

            ps = PoseStamped()
            ps.header.frame_id = map_frame
            ps.pose.position.x = gx
            ps.pose.position.y = gy
            ps.pose.position.z = 0.0
            ps.pose.orientation = _yaw_to_quat(yaw)

            if prefer_origin_xyz is not None:
                cost = float(
                    np.linalg.norm(np.array([gx, gy]) - prefer_origin_xyz[:2])
                )
            else:
                cost = 0.0
            candidates.append((cost, ps))

        if candidates:
            break  # first ring with any success wins

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]
