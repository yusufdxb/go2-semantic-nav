"""Read a GO2 RGB + LiDAR capture folder for the offline calibration tools.

Layout of one segment (``checkerboard``, ``scene`` or ``yaw``)::

    <capture_root>/<segment>/
      meta.yaml        segment, started_utc, camera_frame, intrinsics (ROS ost.yaml dict),
                       extrinsics (parent_frame, child_frame, xyz, rpy, nominal), latency_s,
                       notes, optional fixed_frame (default odom), optional taped_objects
      frames.csv       stamp_ns,file,width,height
      frames/<stamp_ns>.jpg
      clouds.csv       stamp_ns,file,frame_id,n_points
      clouds/<stamp_ns>.npy   Nx3 float32 in frame_id (odom, or the extrinsics parent frame)
      odom.csv         stamp_ns,x,y,z,qx,qy,qz,qw,wz   (pose of the base in odom, base yaw rate)

All stamps are on the robot computer's local clock. Geometry (projection,
depth rendering, the visibility filter) comes from the ``go2_rgb_lidar``
package so these tools see exactly what ``lidar_depth_node`` sees.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_PKG = Path(__file__).resolve().parents[2] / "ros2_ws" / "src" / "go2_rgb_lidar"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from go2_rgb_lidar.calibration import (  # noqa: E402
    CameraModel,
    Extrinsics,
    camera_model_from_dict,
    extrinsics_from_dict,
    load_camera_model,
    load_extrinsics,
)
from go2_rgb_lidar.projection import (  # noqa: E402
    drop_occluded,
    invert_transform,
    make_transform,
    project_points,
    quaternion_to_matrix,
    render_depth_mm,
    rpy_to_matrix,
    transform_points,
)

__all__ = [
    "CameraModel",
    "Extrinsics",
    "Segment",
    "base_from_camera",
    "camera_from_odom",
    "camera_model_from_dict",
    "clouds_between",
    "drop_occluded",
    "extrinsics_from_dict",
    "invert_transform",
    "load_camera_model",
    "load_extrinsics",
    "load_segment",
    "make_transform",
    "matrix_to_rpy",
    "node_depth_mm",
    "odom_pose",
    "project_points",
    "read_image",
    "render_depth_mm",
    "rpy_to_matrix",
    "transform_points",
    "write_yaml",
]

# lidar_depth_node defaults, mirrored so offline results match the robot.
NODE_MIN_DEPTH_M = 0.2
NODE_MAX_DEPTH_M = 8.0
NODE_MAX_NORMALIZED_RADIUS = 1.3
NODE_SPLAT_RADIUS_PX = 2
NODE_OCCLUSION_WINDOW_PX = 21
NODE_OCCLUSION_MARGIN_M = 0.3
NODE_ACCUMULATE_S = 0.5
NODE_FUTURE_TOLERANCE_S = 0.1


@dataclass
class Segment:
    path: Path
    meta: dict
    frame_stamps: np.ndarray  # int64 ns
    frame_files: list[Path]
    frame_size: tuple[int, int]  # (width, height)
    cloud_stamps: np.ndarray  # int64 ns
    cloud_files: list[Path]
    cloud_frames: list[str]
    odom_stamps: np.ndarray  # int64 ns
    odom_values: np.ndarray  # (N, 8): x y z qx qy qz qw wz

    @property
    def fixed_frame(self) -> str:
        return str(self.meta.get("fixed_frame", "odom"))

    @property
    def latency_s(self) -> float:
        return float(self.meta.get("latency_s", 0.0))

    def camera_model(self) -> CameraModel:
        return camera_model_from_dict(self.meta["intrinsics"])

    def extrinsics(self) -> Extrinsics:
        return extrinsics_from_dict(self.meta["extrinsics"])


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_segment(capture_root: str | Path, name: str) -> Segment:
    seg_dir = Path(capture_root) / name
    meta_path = seg_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"no {name!r} segment: {meta_path} missing")
    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    frames = sorted(_read_csv(seg_dir / "frames.csv"), key=lambda r: int(r["stamp_ns"]))
    if not frames:
        raise ValueError(f"{seg_dir}/frames.csv lists no frames")
    sizes = {(int(r["width"]), int(r["height"])) for r in frames}
    if len(sizes) != 1:
        raise ValueError(f"{seg_dir}: frames have mixed sizes {sorted(sizes)}")

    clouds = sorted(_read_csv(seg_dir / "clouds.csv"), key=lambda r: int(r["stamp_ns"]))
    odom = sorted(_read_csv(seg_dir / "odom.csv"), key=lambda r: int(r["stamp_ns"]))
    odom_cols = ("x", "y", "z", "qx", "qy", "qz", "qw", "wz")
    return Segment(
        path=seg_dir,
        meta=meta,
        frame_stamps=np.array([int(r["stamp_ns"]) for r in frames], dtype=np.int64),
        frame_files=[seg_dir / r["file"] for r in frames],
        frame_size=sizes.pop(),
        cloud_stamps=np.array([int(r["stamp_ns"]) for r in clouds], dtype=np.int64),
        cloud_files=[seg_dir / r["file"] for r in clouds],
        cloud_frames=[r["frame_id"] for r in clouds],
        odom_stamps=np.array([int(r["stamp_ns"]) for r in odom], dtype=np.int64),
        odom_values=np.array([[float(r[c]) for c in odom_cols] for r in odom], dtype=np.float64).reshape(-1, 8),
    )


def read_image(path: Path, gray: bool = False) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return img


def odom_pose(seg: Segment, stamp_ns: int, tolerance_s: float = 0.2) -> np.ndarray:
    """Pose of the base in odom (4x4) at ``stamp_ns``, linearly interpolated (quaternion nlerp)."""
    t = seg.odom_stamps
    if t.size == 0:
        raise ValueError(f"{seg.path}: odom.csv is empty")
    tol = int(tolerance_s * 1e9)
    if stamp_ns < t[0] - tol or stamp_ns > t[-1] + tol:
        raise ValueError(f"stamp {stamp_ns} is outside odom coverage [{t[0]}, {t[-1]}] by more than {tolerance_s} s")
    stamp_ns = int(np.clip(stamp_ns, t[0], t[-1]))
    j = int(np.searchsorted(t, stamp_ns))
    if j == 0 or t[j] == stamp_ns or j >= t.size:
        v = seg.odom_values[min(j, t.size - 1)]
    else:
        a = (stamp_ns - t[j - 1]) / float(t[j] - t[j - 1])
        v0, v1 = seg.odom_values[j - 1], seg.odom_values[j]
        q0, q1 = v0[3:7], v1[3:7].copy()
        if np.dot(q0, q1) < 0:
            q1 = -q1
        v = np.concatenate([(1 - a) * v0[:3] + a * v1[:3], (1 - a) * q0 + a * q1, v0[7:]])
    return make_transform(quaternion_to_matrix(*v[3:7]), v[:3])


def base_from_camera(xyz, rpy) -> np.ndarray:
    """Pose of the camera optical frame in the base frame."""
    return make_transform(rpy_to_matrix(*rpy), xyz)


def camera_from_odom(seg: Segment, stamp_ns: int, xyz, rpy) -> np.ndarray:
    return invert_transform(base_from_camera(xyz, rpy)) @ invert_transform(odom_pose(seg, stamp_ns))


def clouds_between(seg: Segment, start_ns: int, end_ns: int) -> np.ndarray:
    """All cloud points stamped in [start_ns, end_ns], in the fixed (odom) frame."""
    base_frame = str(seg.meta.get("extrinsics", {}).get("parent_frame", "base_link"))
    chunks = []
    for stamp, path, frame in zip(seg.cloud_stamps, seg.cloud_files, seg.cloud_frames):
        if not start_ns <= stamp <= end_ns:
            continue
        pts = np.load(path).astype(np.float64).reshape(-1, 3)
        if frame == base_frame:
            pts = transform_points(odom_pose(seg, int(stamp)), pts)
        elif frame != seg.fixed_frame:
            raise ValueError(f"cloud frame {frame!r} is neither {seg.fixed_frame!r} nor {base_frame!r}")
        chunks.append(pts[np.isfinite(pts).all(axis=1)])
    if not chunks:
        return np.empty((0, 3))
    return np.concatenate(chunks, axis=0)


def node_depth_mm(seg: Segment, cam: CameraModel, xyz, rpy, stamp_ns: int) -> np.ndarray:
    """The depth image lidar_depth_node would publish for a frame stamped ``stamp_ns``."""
    pts = clouds_between(
        seg, stamp_ns - int(NODE_ACCUMULATE_S * 1e9), stamp_ns + int(NODE_FUTURE_TOLERANCE_S * 1e9)
    )
    pc = transform_points(camera_from_odom(seg, stamp_ns, xyz, rpy), pts)
    u, v, z = project_points(pc, cam, NODE_MIN_DEPTH_M, NODE_MAX_DEPTH_M, NODE_MAX_NORMALIZED_RADIUS)
    depth = render_depth_mm(u, v, z, cam.width, cam.height, NODE_SPLAT_RADIUS_PX)
    return drop_occluded(depth, NODE_OCCLUSION_WINDOW_PX, NODE_OCCLUSION_MARGIN_M)


def matrix_to_rpy(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_to_matrix (URDF fixed-axis: R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    roll = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def write_yaml(path: str | Path, data: dict, header: str = "") -> None:
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("".join(f"# {line}\n" for line in header.splitlines()))
        f.write(text)
