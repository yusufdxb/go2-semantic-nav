"""Unit tests for LiDAR-to-camera projection and sparse depth rendering."""

import cv2
import numpy as np
import pytest
from go2_open_vocab_detector.depth_backproject import CameraIntrinsics, object_centroid_3d
from go2_rgb_lidar.calibration import camera_model_from_dict
from go2_rgb_lidar.projection import (
    CloudWindow,
    drop_occluded,
    invert_transform,
    make_transform,
    project_points,
    quaternion_to_matrix,
    render_depth_mm,
    rpy_to_matrix,
    transform_points,
)


def _cam(d=(0.0, 0.0, 0.0, 0.0, 0.0), width=640, height=480):
    return camera_model_from_dict(
        {
            "image_width": width,
            "image_height": height,
            "camera_matrix": {"rows": 3, "cols": 3, "data": [500, 0, 320, 0, 500, 240, 0, 0, 1]},
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {"rows": 1, "cols": 5, "data": list(d)},
        }
    )


def test_rpy_matches_quaternion_for_optical_frame():
    # Optical frame (z fwd, x right, y down) seen from a body frame (x fwd, y left, z up).
    r_rpy = rpy_to_matrix(-np.pi / 2, 0.0, -np.pi / 2)
    r_q = quaternion_to_matrix(-0.5, 0.5, -0.5, 0.5)
    np.testing.assert_allclose(r_rpy, r_q, atol=1e-12)
    # Optical +z must be body +x, optical +x body -y, optical +y body -z.
    np.testing.assert_allclose(r_rpy @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(r_rpy @ [1, 0, 0], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(r_rpy @ [0, 1, 0], [0, 0, -1], atol=1e-12)


def test_invert_transform_round_trip():
    t = make_transform(rpy_to_matrix(0.1, -0.2, 0.3), [1.0, -2.0, 0.5])
    pts = np.random.default_rng(0).normal(size=(50, 3))
    np.testing.assert_allclose(transform_points(invert_transform(t), transform_points(t, pts)), pts, atol=1e-12)


def test_on_axis_point_hits_principal_point():
    u, v, z = project_points(np.array([[0.0, 0.0, 2.0]]), _cam(), 0.2, 8.0, 2.0)
    assert (u[0], v[0], z[0]) == (320.0, 240.0, 2.0)


@pytest.mark.parametrize("d", [(0.0, 0.0, 0.0, 0.0, 0.0), (-0.28, 0.07, 0.001, -0.0005, -0.008)])
def test_projection_matches_opencv(d):
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-1.5, 1.5, 400), rng.uniform(-1.0, 1.0, 400), rng.uniform(1.0, 6.0, 400)])
    cam = _cam(d)
    u, v, z = project_points(pts, cam, 0.2, 8.0, 10.0)
    ref, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), cam.k, cam.d)
    ref = ref.reshape(-1, 2)
    inside = (ref[:, 0] > -0.5) & (ref[:, 0] < 639.5) & (ref[:, 1] > -0.5) & (ref[:, 1] < 479.5)
    np.testing.assert_allclose(np.column_stack([u, v]), ref[inside], atol=1e-6)
    np.testing.assert_allclose(z, pts[inside, 2])


def test_points_behind_out_of_range_or_off_image_are_dropped():
    pts = np.array(
        [
            [0.0, 0.0, -1.0],  # behind the camera
            [0.0, 0.0, 0.1],  # closer than min depth
            [0.0, 0.0, 9.0],  # beyond max depth
            [10.0, 0.0, 1.0],  # far off to the side
            [np.nan, 0.0, 2.0],
            [0.1, 0.1, 2.0],  # the only valid one
        ]
    )
    u, v, z = project_points(pts, _cam(), 0.2, 8.0, 2.0)
    assert z.tolist() == [2.0]


def test_strong_distortion_cannot_fold_far_points_back_into_image():
    # With k1 strongly negative the polynomial turns over: a point at a steep
    # angle would land near the image centre. The radius guard must drop it.
    cam = _cam((-0.6, 0.0, 0.0, 0.0, 0.0))
    # At normalized radius 1.265 the radial factor 1 - 0.6 r^2 is ~0.04.
    steep = np.array([[1.265, 0.0, 1.0]])
    u, _, _ = project_points(steep, cam, 0.2, 8.0, 10.0)
    assert u.size == 1 and abs(u[0] - 320) < 40  # unguarded: folds back near the centre
    u, _, _ = project_points(steep, cam, 0.2, 8.0, 1.2)
    assert u.size == 0


def test_render_nearest_depth_wins_and_zero_means_no_return():
    u = np.array([10.0, 10.0, 30.0])
    v = np.array([20.0, 20.0, 5.0])
    z = np.array([3.0, 1.25, 2.0])
    depth = render_depth_mm(u, v, z, 64, 48, splat_radius_px=0)
    assert depth.dtype == np.uint16
    assert depth[20, 10] == 1250
    assert depth[5, 30] == 2000
    assert np.count_nonzero(depth) == 2


def test_render_splat_covers_square_and_clips_at_border():
    depth = render_depth_mm(np.array([0.0]), np.array([0.0]), np.array([1.0]), 10, 10, splat_radius_px=2)
    assert np.count_nonzero(depth) == 9  # 3x3 survives of the 5x5 at the corner
    assert np.all(depth[:3, :3] == 1000)


def test_lidar_depth_feeds_detector_backprojection():
    """A board sampled like LiDAR rings, with background leaking through the gaps, round-trips to its 3D centre."""
    cam = _cam()
    # A 0.6 m x 0.6 m flat board centred 0.3 m right, 0.1 m down, 2.5 m ahead,
    # plus a wall 6 m away that the rings also hit through the gaps.
    ys = np.linspace(-0.2, 0.4, 7)  # sparse rows, like LiDAR rings
    xs = np.linspace(0.0, 0.6, 40)
    board = np.array([[x, y, 2.5] for y in ys for x in xs])
    wall = np.array([[x, y, 6.0] for y in np.linspace(-1, 1, 15) for x in np.linspace(-2, 2, 60)])
    u, v, z = project_points(np.vstack([board, wall]), cam, 0.2, 8.0, 2.0)
    depth = render_depth_mm(u, v, z, cam.width, cam.height, splat_radius_px=2)
    # Without the visibility filter the wall seen between the rings wins the median.
    depth = drop_occluded(depth, window_px=21, margin_m=0.3)

    # The segmenter's mask for the board, from its true outline.
    mask = np.zeros((cam.height, cam.width), dtype=bool)
    u0, v0 = 500 * 0.0 / 2.5 + 320, 500 * -0.2 / 2.5 + 240
    u1, v1 = 500 * 0.6 / 2.5 + 320, 500 * 0.4 / 2.5 + 240
    mask[int(v0) : int(v1) + 1, int(u0) : int(u1) + 1] = True

    intr = CameraIntrinsics(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, width=cam.width, height=cam.height)
    centroid, depth_m, _ = object_centroid_3d(mask, depth, intr)
    assert depth_m == pytest.approx(2.5, abs=0.002)
    np.testing.assert_allclose(centroid, [0.3, 0.1, 2.5], atol=0.02)


def test_cloud_window_keeps_horizon_and_selects_by_time():
    w = CloudWindow(horizon_s=0.5)
    for i in range(10):  # 10 clouds at 10 Hz
        w.add(int(i * 1e8), np.full((2, 3), float(i), dtype=np.float32))
    assert len(w) == 6  # stamps 0.4 .. 0.9 s are within 0.5 s of the newest
    assert w.nearest_gap_s(int(0.93e9)) == pytest.approx(0.03)
    pts = w.points_between(int(0.6e9), int(0.8e9))
    assert sorted(set(pts[:, 0].tolist())) == [6.0, 7.0, 8.0]
    assert w.points_between(int(5e9), int(6e9)).shape == (0, 3)
    assert CloudWindow(0.5).nearest_gap_s(0) == float("inf")


def test_drop_occluded_keeps_foreground_and_far_background():
    depth = np.zeros((40, 40), dtype=np.uint16)
    depth[10, 10] = 2000  # foreground return
    depth[10, 14] = 6000  # background 4 px away: hidden
    depth[10, 30] = 6000  # background 20 px away: outside the window, kept
    depth[12, 12] = 2200  # within the margin of the foreground: kept
    out = drop_occluded(depth, window_px=11, margin_m=0.3)
    assert out[10, 10] == 2000 and out[12, 12] == 2200 and out[10, 30] == 6000
    assert out[10, 14] == 0
    assert drop_occluded(depth, window_px=1, margin_m=0.3) is depth
