"""Sample a reachable stand-off pose around a target object, optionally gated by a costmap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid


@dataclass
class SamplerParams:
    stand_off_m: float = 0.9
    goal_ring_samples: int = 12
    goal_min_stand_off_m: float = 0.4
    goal_max_stand_off_m: float = 2.0
    costmap_free_max_cost: int = 50
    retry_with_wider_ring: bool = True
    allow_unknown: bool = True


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


def _is_free(costmap: Optional[OccupancyGrid], x: float, y: float, max_cost: int,
             allow_unknown: bool = False) -> bool:
    return _cell_state(costmap, x, y, max_cost, allow_unknown) is not None


def _cell_state(costmap: Optional[OccupancyGrid], x: float, y: float, max_cost: int,
                allow_unknown: bool) -> Optional[str]:
    """'free', 'unknown' (only if allowed), or None when the cell is not usable."""
    if costmap is None:
        return "free"  # no costmap → don't block; caller is responsible
    pt = _world_to_grid(x, y, costmap)
    if pt is None:
        return None  # off-map
    cost = _cell_cost(costmap, *pt)
    if cost < 0:
        # Unknown. Objects are often seen by the camera before the LiDAR has
        # mapped the room they are in; Nav2 plans through unknown space and
        # replans as it is observed. Known-free slots are still preferred.
        return "unknown" if allow_unknown else None
    return "free" if cost <= max_cost else None


def sample_stand_off(
    target_xyz: np.ndarray,
    costmap: Optional[OccupancyGrid],
    params: SamplerParams | None = None,
    prefer_origin_xyz: Optional[np.ndarray] = None,
    stand_off_override: Optional[float] = None,
    map_frame: str = "map",
    target_half_extent_m: float = 0.0,
) -> Optional[PoseStamped]:
    """Return a PoseStamped standing `stand_off_m` from the target, facing it,
    landing on a free-ish costmap cell. Returns None if no sample passes.

    ``stand_off_m`` is measured from the object's SURFACE, approximated as a
    circle of radius ``target_half_extent_m`` around its centroid. Measuring
    from the centroid put the robot's nose inside the LiDAR hazard stop
    distance of any object larger than ~0.3 m, so every arrival ended in a
    latched emergency stop.

    ``prefer_origin_xyz`` is the robot's current pose (if known); sampled poses
    are ranked by closeness to it so the robot picks the nearest reachable slot.
    """
    p = params or SamplerParams()
    surface = stand_off_override or p.stand_off_m
    surface = max(p.goal_min_stand_off_m, min(p.goal_max_stand_off_m, surface))
    half_extent = max(0.0, float(target_half_extent_m))

    candidates: list[tuple[float, PoseStamped]] = []

    # Two sweeps: primary ring at the requested surface distance, secondary at
    # 1.3x (clamped) if needed. Both are offset by the object's half extent.
    surfaces = [surface]
    if p.retry_with_wider_ring:
        surfaces.append(min(surface * 1.3, p.goal_max_stand_off_m))
    # Deduplicate if both rings landed on same radius.
    rings = list(dict.fromkeys(round(s_ + half_extent, 3) for s_ in surfaces))

    for ring_radius in rings:
        for k in range(p.goal_ring_samples):
            theta = 2.0 * math.pi * k / p.goal_ring_samples
            gx = float(target_xyz[0] + ring_radius * math.cos(theta))
            gy = float(target_xyz[1] + ring_radius * math.sin(theta))
            state = _cell_state(costmap, gx, gy, p.costmap_free_max_cost, p.allow_unknown)
            if state is None:
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
            if state == "unknown":
                cost += 1.0e3  # any known-free slot beats any unknown one
            candidates.append((cost, ps))

        if candidates:
            break  # first ring with any success wins

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]
