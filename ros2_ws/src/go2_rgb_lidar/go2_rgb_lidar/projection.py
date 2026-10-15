"""Project LiDAR points into the front camera and render a sparse aligned depth image.

The output is a uint16 depth image in millimetres with 0 = no return, the same
convention as a RealSense aligned-depth stream, so the open-vocab detector's
masked-median back-projection consumes it unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .calibration import CameraModel


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF/tf2 fixed-axis convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = np.asarray(translation, dtype=np.float64)
    return t


def invert_transform(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    return make_transform(r.T, -r.T @ t[:3, 3])


def transform_points(t: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an Nx3 array."""
    return points @ t[:3, :3].T + t[:3, 3]


def project_points(
    points_cam: np.ndarray,
    cam: CameraModel,
    min_depth_m: float,
    max_depth_m: float,
    max_normalized_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-optical-frame points (z forward) to raw-image pixels.

    Applies plumb_bob distortion so points land where the camera saw them.
    Points outside [min_depth_m, max_depth_m], outside the image, or beyond
    ``max_normalized_radius`` (where the distortion polynomial is no longer
    trustworthy and can fold back into the image) are dropped.
    Returns (u, v, z) for the kept points; u, v are pixel coordinates with
    pixel centres at integers.
    """
    if points_cam.size == 0:
        empty = np.empty(0)
        return empty, empty, empty
    z = points_cam[:, 2]
    keep = np.isfinite(points_cam).all(axis=1) & (z > min_depth_m) & (z < max_depth_m)
    pts = points_cam[keep]
    z = pts[:, 2]
    x = pts[:, 0] / z
    y = pts[:, 1] / z
    r2 = x * x + y * y
    keep = r2 <= max_normalized_radius**2
    x, y, z, r2 = x[keep], y[keep], z[keep], r2[keep]
    k1, k2, p1, p2, k3 = cam.d
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    u = cam.fx * xd + cam.cx
    v = cam.fy * yd + cam.cy
    inside = (u > -0.5) & (u < cam.width - 0.5) & (v > -0.5) & (v < cam.height - 0.5)
    return u[inside], v[inside], z[inside]


def render_depth_mm(
    u: np.ndarray,
    v: np.ndarray,
    z_m: np.ndarray,
    width: int,
    height: int,
    splat_radius_px: int = 0,
) -> np.ndarray:
    """Rasterise projected points into a uint16 mm depth image, 0 = no return.

    Each point covers a (2r+1)^2 square; where squares overlap the NEAREST
    depth wins, so a foreground object is not filled in by the background
    seen through the gaps between LiDAR rings.
    """
    depth = np.full(height * width, np.iinfo(np.uint16).max, dtype=np.uint16)
    if u.size:
        z_mm = np.clip(np.rint(z_m * 1000.0), 1, np.iinfo(np.uint16).max - 1).astype(np.uint16)
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        r = int(max(0, splat_radius_px))
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                uu, vv = ui + du, vi + dv
                ok = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                np.minimum.at(depth, vv[ok] * width + uu[ok], z_mm[ok])
    depth[depth == np.iinfo(np.uint16).max] = 0
    return depth.reshape(height, width)


def drop_occluded(depth_mm: np.ndarray, window_px: int, margin_m: float) -> np.ndarray:
    """Zero out returns that have a much nearer return within a window around them.

    The LiDAR sits below the camera and the window accumulates clouds while
    the robot moves, so background points the camera cannot see show through
    the gaps between LiDAR rings on a foreground object. Left in, they can
    outnumber the object's own returns and drag the detector's masked median
    to the background. This is the standard visibility filter for projected
    LiDAR: a return farther than ``margin_m`` behind the nearest return within
    ``window_px`` is treated as hidden. It also thins the background for
    ``window_px / 2`` pixels around every object edge.
    """
    if window_px <= 1:
        return depth_mm
    import cv2  # local: the pure-math functions above need no OpenCV

    d = depth_mm.astype(np.float32)
    d[depth_mm == 0] = np.inf
    nearest = cv2.erode(d, np.ones((window_px, window_px), np.uint8), borderType=cv2.BORDER_REPLICATE)
    hidden = (depth_mm > 0) & (d > nearest + margin_m * 1000.0)
    out = depth_mm.copy()
    out[hidden] = 0
    return out


class CloudWindow:
    """Short history of LiDAR clouds already expressed in one fixed frame.

    A single GO2 LiDAR sweep is sparse in the camera view; accumulating a few
    hundred milliseconds of clouds in a world-fixed frame (odom) densifies the
    depth image without smearing static structure while the robot moves.
    Moving objects (people) do smear by their own motion over the window.
    """

    def __init__(self, horizon_s: float) -> None:
        self._horizon_ns = int(horizon_s * 1e9)
        self._clouds: deque[tuple[int, np.ndarray]] = deque()

    def add(self, stamp_ns: int, points: np.ndarray) -> None:
        self._clouds.append((stamp_ns, points))
        newest = max(s for s, _ in self._clouds)
        while self._clouds and self._clouds[0][0] < newest - self._horizon_ns:
            self._clouds.popleft()

    def __len__(self) -> int:
        return len(self._clouds)

    def nearest_gap_s(self, stamp_ns: int) -> float:
        """|cloud stamp - stamp| of the closest cloud in time; inf if empty."""
        if not self._clouds:
            return float("inf")
        return min(abs(s - stamp_ns) for s, _ in self._clouds) / 1e9

    def points_between(self, start_ns: int, end_ns: int) -> np.ndarray:
        chunks = [p for s, p in self._clouds if start_ns <= s <= end_ns]
        if not chunks:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(chunks, axis=0)
