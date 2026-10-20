"""Front-camera calibration files: intrinsics (ROS camera_calibration YAML) and extrinsics.

Both files may carry a top-level ``nominal: true`` key. A nominal file holds
datasheet or guessed values, not a measurement; the depth node refuses to
publish depth from one unless explicitly allowed, because a wrong intrinsic or
extrinsic places every detected object at the wrong 3D point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

SUPPORTED_DISTORTION_MODELS = ("plumb_bob",)


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    k: np.ndarray  # 3x3
    d: np.ndarray  # (5,) plumb_bob k1 k2 p1 p2 k3
    p: np.ndarray  # 3x4
    distortion_model: str
    nominal: bool

    @property
    def fx(self) -> float:
        return float(self.k[0, 0])

    @property
    def fy(self) -> float:
        return float(self.k[1, 1])

    @property
    def cx(self) -> float:
        return float(self.k[0, 2])

    @property
    def cy(self) -> float:
        return float(self.k[1, 2])


@dataclass(frozen=True)
class Extrinsics:
    parent_frame: str
    child_frame: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    nominal: bool


def _matrix(entry: dict, rows: int, cols: int, key: str) -> np.ndarray:
    data = entry.get("data") if isinstance(entry, dict) else None
    if data is None or len(data) != rows * cols:
        raise ValueError(f"{key} must have {rows * cols} values in 'data'")
    return np.asarray(data, dtype=np.float64).reshape(rows, cols)


def camera_model_from_dict(cfg: dict) -> CameraModel:
    width = int(cfg["image_width"])
    height = int(cfg["image_height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    k = _matrix(cfg["camera_matrix"], 3, 3, "camera_matrix")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("camera_matrix fx and fy must be positive")
    model = str(cfg.get("distortion_model", "plumb_bob"))
    if model not in SUPPORTED_DISTORTION_MODELS:
        raise ValueError(f"distortion_model {model!r} unsupported; use one of {SUPPORTED_DISTORTION_MODELS}")
    d = np.asarray(cfg.get("distortion_coefficients", {}).get("data", [0.0] * 5), dtype=np.float64)
    if d.shape != (5,):
        raise ValueError(f"plumb_bob needs 5 distortion coefficients, got {d.size}")
    if "projection_matrix" in cfg:
        p = _matrix(cfg["projection_matrix"], 3, 4, "projection_matrix")
    else:
        p = np.hstack([k, np.zeros((3, 1))])
    return CameraModel(
        width=width,
        height=height,
        k=k,
        d=d,
        p=p,
        distortion_model=model,
        nominal=bool(cfg.get("nominal", False)),
    )


def load_camera_model(path: str) -> CameraModel:
    with open(path, encoding="utf-8") as f:
        return camera_model_from_dict(yaml.safe_load(f))


def extrinsics_from_dict(cfg: dict) -> Extrinsics:
    xyz = tuple(float(v) for v in cfg["xyz"])
    rpy = tuple(float(v) for v in cfg["rpy"])
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError("extrinsics xyz and rpy must each have 3 values")
    parent = str(cfg["parent_frame"])
    child = str(cfg["child_frame"])
    if not parent or not child or parent == child:
        raise ValueError("extrinsics need distinct, non-empty parent_frame and child_frame")
    return Extrinsics(parent, child, xyz, rpy, bool(cfg.get("nominal", False)))


def load_extrinsics(path: str) -> Extrinsics:
    with open(path, encoding="utf-8") as f:
        return extrinsics_from_dict(yaml.safe_load(f))
