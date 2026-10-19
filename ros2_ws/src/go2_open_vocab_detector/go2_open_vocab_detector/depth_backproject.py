"""Depth back-projection utilities.

Takes a depth image (mm, uint16, from RealSense) plus a camera intrinsics matrix
and converts per-object 2D masks into 3D centroids in the camera optical frame.
Units: output is meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rgbd_alignment_error(
    *,
    color_frame: str,
    depth_frame: str,
    info_frame: str,
    color_shape: tuple[int, int],
    depth_shape: tuple[int, int],
    info_shape: tuple[int, int],
) -> str | None:
    """Return an error unless depth and intrinsics are aligned to the color image."""
    if not color_frame or depth_frame != color_frame or info_frame != color_frame:
        return (
            "RGB, depth, and camera_info must share the color optical frame; "
            f"got color={color_frame!r}, depth={depth_frame!r}, info={info_frame!r}"
        )
    if depth_shape != color_shape or info_shape != color_shape:
        return (
            "RGB, depth, and camera_info dimensions must match; "
            f"got color={color_shape}, depth={depth_shape}, info={info_shape}"
        )
    return None


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # plumb_bob (k1, k2, p1, p2, k3) of the raw image; empty means a pinhole (undistorted) image
    distortion: tuple[float, ...] = ()

    @classmethod
    def from_camera_info_k(
        cls, k: list[float], width: int, height: int, d: list[float] | None = None
    ) -> "CameraIntrinsics":
        # k is row-major [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        dist = tuple(float(v) for v in d[:5]) if d is not None and len(d) >= 5 and any(d[:5]) else ()
        return cls(fx=k[0], fy=k[4], cx=k[2], cy=k[5], width=width, height=height, distortion=dist)


def backproject_pixel(u: float, v: float, depth_m: float, intr: CameraIntrinsics) -> np.ndarray:
    """Back-project a single raw-image pixel + metric depth to a 3D point in the camera optical frame.

    ``depth_m`` is the optical-axis depth. Lens distortion (plumb_bob) is undone by fixed-point iteration, the
    same scheme as OpenCV's undistortPoints, so this inverts ``go2_rgb_lidar.projection.project_points``.
    """
    xd = (u - intr.cx) / intr.fx
    yd = (v - intr.cy) / intr.fy
    x, y = xd, yd
    if intr.distortion:
        k1, k2, p1, p2, k3 = intr.distortion
        for _ in range(25):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x = (xd - (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))) / radial
            y = (yd - (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)) / radial
    return np.array([x * depth_m, y * depth_m, depth_m], dtype=np.float32)


def masked_depth_median(
    depth_image_mm: np.ndarray,
    mask: np.ndarray,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_valid_pixels: int = 1,
) -> float:
    """Robust median depth over a binary mask, in meters.

    Rejects zeros (RealSense invalid-depth sentinel) and out-of-range values.
    Returns NaN if fewer than ``min_valid_pixels`` valid pixels remain. A dense
    RGB-D frame needs 1; a sparse LiDAR-projected frame should demand several
    returns so one stray point cannot place the object.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if depth_image_mm.shape != mask.shape:
        raise ValueError(
            f"depth/mask shape mismatch: {depth_image_mm.shape} vs {mask.shape}"
        )
    valid_mask = mask & (depth_image_mm > int(min_depth_m * 1000)) & (
        depth_image_mm < int(max_depth_m * 1000)
    )
    if np.count_nonzero(valid_mask) < max(1, int(min_valid_pixels)):
        return float("nan")
    depths_m = depth_image_mm[valid_mask].astype(np.float32) / 1000.0
    return float(np.median(depths_m))


def object_centroid_3d(
    mask: np.ndarray,
    depth_image_mm: np.ndarray,
    intr: CameraIntrinsics,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_valid_pixels: int = 1,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute (centroid_3d_camera, depth_median_m, dimensions_xyz) for a mask.

    - centroid_3d_camera: np.array([x, y, z]) in meters, camera optical frame.
    - depth_median_m: robust depth, or NaN on failure.
    - dimensions_xyz: crude (dx, dy, dz) bounding extents in meters derived from
      the back-projected 2D bbox corners at the object's depth. Coarse but useful
      for RViz markers and scene-graph sizing.
    Returns (NaN array, NaN, NaN array) if the mask has no valid depth.
    """
    depth_m = masked_depth_median(
        depth_image_mm,
        mask,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        min_valid_pixels=min_valid_pixels,
    )
    if not np.isfinite(depth_m):
        nan3 = np.full(3, np.nan, dtype=np.float32)
        return nan3, float("nan"), nan3.copy()

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        nan3 = np.full(3, np.nan, dtype=np.float32)
        return nan3, float("nan"), nan3.copy()

    u_mean = float(np.mean(xs))
    v_mean = float(np.mean(ys))
    centroid = backproject_pixel(u_mean, v_mean, depth_m, intr)

    # Pixel indices are pixel centres; a mask covering columns [u_min, u_max]
    # spans (u_max - u_min + 1) pixels, from edge u_min - 0.5 to edge u_max + 0.5.
    u_min, u_max = float(np.min(xs)) - 0.5, float(np.max(xs)) + 0.5
    v_min, v_max = float(np.min(ys)) - 0.5, float(np.max(ys)) + 0.5
    p_min = backproject_pixel(u_min, v_min, depth_m, intr)
    p_max = backproject_pixel(u_max, v_max, depth_m, intr)
    dims_xy = np.abs(p_max - p_min)
    # Z extent is not observable from a single mask at one depth; approximate as dy.
    dims_xyz = np.array([dims_xy[0], dims_xy[1], dims_xy[1]], dtype=np.float32)

    return centroid.astype(np.float32), float(depth_m), dims_xyz


def mask_rle_encode(mask: np.ndarray) -> list[int]:
    """Row-major run-length encoding of a binary mask.

    Output format: [count_of_zeros, count_of_ones, count_of_zeros, ...],
    starting from the first pixel. Simple and compatible with COCO-style RLE
    consumers when paired with the mask dimensions (carried in mask_roi).
    """
    flat = mask.astype(bool).ravel(order="C").astype(np.uint8)
    if flat.size == 0:
        return []
    # Find boundaries
    diffs = np.diff(flat.astype(np.int16))
    change_indices = np.where(diffs != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], change_indices, [flat.size])))
    # If the mask starts with a 1, prepend a zero-length run of zeros so the
    # RLE always starts with a zero-run count.
    if flat[0] == 1:
        lengths = np.concatenate(([0], lengths))
    return lengths.astype(np.int32).tolist()


def mask_rle_decode(rle: list[int], height: int, width: int) -> np.ndarray:
    """Inverse of mask_rle_encode."""
    total = height * width
    out = np.zeros(total, dtype=np.uint8)
    cursor = 0
    value = 0
    for count in rle:
        if count < 0:
            raise ValueError("RLE counts must be non-negative")
        end = cursor + count
        if end > total:
            raise ValueError("RLE overruns mask extent")
        if value == 1:
            out[cursor:end] = 1
        cursor = end
        value = 1 - value
    if cursor != total:
        raise ValueError(f"RLE underruns mask extent: {cursor} != {total}")
    return out.reshape((height, width)).astype(bool)
