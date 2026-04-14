"""Depth back-projection utilities.

Takes a depth image (mm, uint16, from RealSense) plus a camera intrinsics matrix
and converts per-object 2D masks into 3D centroids in the camera optical frame.
Units: output is meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info_k(cls, k: list[float], width: int, height: int) -> "CameraIntrinsics":
        # k is row-major [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        return cls(fx=k[0], fy=k[4], cx=k[2], cy=k[5], width=width, height=height)


def backproject_pixel(u: float, v: float, depth_m: float, intr: CameraIntrinsics) -> np.ndarray:
    """Back-project a single pixel + metric depth to a 3D point in the camera optical frame."""
    x = (u - intr.cx) * depth_m / intr.fx
    y = (v - intr.cy) * depth_m / intr.fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


def masked_depth_median(
    depth_image_mm: np.ndarray,
    mask: np.ndarray,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> float:
    """Robust median depth over a binary mask, in meters.

    Rejects zeros (RealSense invalid-depth sentinel) and out-of-range values.
    Returns NaN if the valid subset is empty.
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
    if not np.any(valid_mask):
        return float("nan")
    depths_m = depth_image_mm[valid_mask].astype(np.float32) / 1000.0
    return float(np.median(depths_m))


def object_centroid_3d(
    mask: np.ndarray,
    depth_image_mm: np.ndarray,
    intr: CameraIntrinsics,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
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
        depth_image_mm, mask, min_depth_m=min_depth_m, max_depth_m=max_depth_m
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

    u_min, u_max = float(np.min(xs)), float(np.max(xs))
    v_min, v_max = float(np.min(ys)), float(np.max(ys))
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
