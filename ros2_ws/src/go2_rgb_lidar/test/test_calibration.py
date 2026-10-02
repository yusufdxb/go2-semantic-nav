"""Calibration file loading, including the shipped nominal files."""

import os

import numpy as np
import pytest
from go2_rgb_lidar.calibration import (
    camera_model_from_dict,
    extrinsics_from_dict,
    load_camera_model,
    load_extrinsics,
)

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")


def _intr(**over):
    cfg = {
        "image_width": 1280,
        "image_height": 720,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [600, 0, 640, 0, 600, 360, 0, 0, 1]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.1, 0, 0, 0, 0]},
    }
    cfg.update(over)
    return cfg


def test_camera_model_fields_and_default_projection():
    cam = camera_model_from_dict(_intr())
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (600, 600, 640, 360)
    np.testing.assert_allclose(cam.p[:, :3], cam.k)
    assert cam.nominal is False


@pytest.mark.parametrize(
    "over",
    [
        {"distortion_model": "equidistant"},
        {"distortion_coefficients": {"data": [0.1, 0.2]}},
        {"camera_matrix": {"data": [0, 0, 640, 0, 600, 360, 0, 0, 1]}},
        {"camera_matrix": {"data": [600, 0, 640]}},
        {"image_width": 0},
    ],
)
def test_camera_model_rejects_bad_files(over):
    with pytest.raises(ValueError):
        camera_model_from_dict(_intr(**over))


def test_extrinsics_reject_same_frames():
    with pytest.raises(ValueError):
        extrinsics_from_dict({"parent_frame": "a", "child_frame": "a", "xyz": [0, 0, 0], "rpy": [0, 0, 0]})


def test_shipped_files_load_and_are_marked_nominal():
    cam = load_camera_model(os.path.join(CONFIG, "front_camera_intrinsics_nominal.yaml"))
    ext = load_extrinsics(os.path.join(CONFIG, "front_camera_extrinsics_nominal.yaml"))
    assert cam.nominal and ext.nominal
    assert (cam.width, cam.height) == (1280, 720)
    assert ext.parent_frame == "base_link"
    assert ext.child_frame == "front_camera_optical_frame"
