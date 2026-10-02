"""extrinsics_from_capture and acceptance_from_capture on a ray-cast box scene."""

import math

import acceptance_from_capture as acc
import capture_io as cio
import extrinsics_from_capture as exf
import numpy as np
import pytest
import synth_capture as sc

CAM = sc.intrinsics_dict(640, 360, 320.0, 320.0, 319.5, 179.5, d=(-0.05, 0.01, 0.0, 0.0, 0.0))
TRUE_XYZ = [0.32, 0.0, 0.03]
TRUE_RPY = [sc.OPTICAL_RPY[0], math.radians(-2.0), sc.OPTICAL_RPY[2]]  # slight in-plane tilt
TAPED = [
    {"label": "box_a", "x_m": 1.8, "y_m": -0.4, "z_m": 0.0},  # front face centre
    {"label": "box_b", "x_m": 2.6, "y_m": 0.65, "z_m": 0.15},
]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("scene")
    init_rpy = list(np.array(TRUE_RPY) + np.radians([2.0, -3.0, 1.5]))
    init_xyz = list(np.array(TRUE_XYZ) + [0.01, -0.01, 0.01])
    sc.scene_capture(root, TRUE_XYZ, TRUE_RPY, init_xyz, init_rpy, CAM, taped=TAPED)
    cio.write_yaml(root / "intr.yaml", CAM)
    cio.write_yaml(root / "true_ext.yaml", sc.extrinsics_dict(TRUE_XYZ, TRUE_RPY))
    return root


def test_extrinsics_recovers_known_offset(scene):
    out = scene / "ext.yaml"
    rc = exf.main(["--capture", str(scene), "--intrinsics", str(scene / "intr.yaml"), "--out", str(out),
                   "--frames", "1", "--overlay-dir", str(scene / "overlays")])
    assert rc == 0
    ext = cio.load_extrinsics(str(out))
    assert ext.nominal is False
    np.testing.assert_allclose(np.degrees(ext.rpy), np.degrees(TRUE_RPY), atol=0.5)
    np.testing.assert_allclose(ext.xyz, TRUE_XYZ, atol=0.02)
    assert (scene / "overlays" / "overlay_after.png").exists()


def test_extrinsics_already_optimal_returns_3(scene):
    out = scene / "ext_from_true.yaml"
    rc = exf.main(["--capture", str(scene), "--intrinsics", str(scene / "intr.yaml"), "--out", str(out),
                   "--init-extrinsics", str(scene / "true_ext.yaml"), "--frames", "1"])
    assert rc == 3 and out.exists()


def test_extrinsics_boundary_fails_and_writes_nothing(scene):
    far = list(TRUE_RPY)
    far[2] += math.radians(9.0)  # 9 deg yaw error: the optimum is outside the +/-5 deg box
    cio.write_yaml(scene / "far_ext.yaml", sc.extrinsics_dict(TRUE_XYZ, far))
    out = scene / "ext_far.yaml"
    rc = exf.main(["--capture", str(scene), "--intrinsics", str(scene / "intr.yaml"), "--out", str(out),
                   "--init-extrinsics", str(scene / "far_ext.yaml"), "--frames", "1"])
    assert rc == 2 and not out.exists()


def _acceptance(scene, ext_file, *extra):
    return acc.main(["--capture", str(scene), "--intrinsics", str(scene / "intr.yaml"),
                     "--extrinsics", str(ext_file), "--frames", "2", *extra])


def test_acceptance_passes_with_true_calibration(scene):
    assert _acceptance(scene, scene / "true_ext.yaml") == 0


def test_acceptance_fails_on_camera_misalignment(scene, capsys):
    bad = list(TRUE_RPY)
    bad[2] += math.radians(3.0)
    cio.write_yaml(scene / "bad_ext.yaml", sc.extrinsics_dict(TRUE_XYZ, bad))
    assert _acceptance(scene, scene / "bad_ext.yaml") == 2
    out = capsys.readouterr().out
    assert "FAIL edge alignment" in out
    # The range check alone cannot see a camera error: it still passes per object.
    assert out.count("ok   box_") == 2
    assert _acceptance(scene, scene / "bad_ext.yaml", "--skip-alignment") == 0


def test_acceptance_fails_on_wrong_tape(tmp_path):
    taped = [dict(TAPED[0], x_m=1.55), TAPED[1]]  # tape 25 cm short of box A
    sc.scene_capture(tmp_path, TRUE_XYZ, TRUE_RPY, TRUE_XYZ, TRUE_RPY, CAM, taped=taped)
    cio.write_yaml(tmp_path / "intr.yaml", CAM)
    cio.write_yaml(tmp_path / "ext.yaml", sc.extrinsics_dict(TRUE_XYZ, TRUE_RPY))
    assert acc.main(["--capture", str(tmp_path), "--intrinsics", str(tmp_path / "intr.yaml"),
                     "--extrinsics", str(tmp_path / "ext.yaml"), "--frames", "2"]) == 2


def test_capture_io_odom_interpolation_and_base_frame_clouds(tmp_path):
    t0 = 10**12
    odom = [(t0, sc.pose(0.0, 0.0, 0.0), 0.0), (t0 + 10**9, sc.pose(1.0, 0.0, math.pi / 2), 0.0)]
    meta = {"intrinsics": CAM, "extrinsics": sc.extrinsics_dict(TRUE_XYZ, TRUE_RPY)}
    pts = np.array([[1.0, 0.0, 0.0]])
    sc.write_segment(tmp_path, "scene", meta, [(t0, np.zeros((360, 640, 3), np.uint8))],
                     [(t0 + 10**9, pts, "base_link"), (t0 + 10**9, pts, "odom")], odom)
    seg = cio.load_segment(tmp_path, "scene")
    mid = cio.odom_pose(seg, t0 + 5 * 10**8)
    np.testing.assert_allclose(mid[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)
    assert math.degrees(cio.matrix_to_rpy(mid[:3, :3])[2]) == pytest.approx(45.0, abs=0.01)
    both = cio.clouds_between(seg, t0, t0 + 10**9)
    # base_link cloud moved to odom with the pose at its stamp (x=1, yaw 90 deg): (1, 1, 0)
    np.testing.assert_allclose(sorted(both.tolist()), [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], atol=1e-6)
    with pytest.raises(ValueError):
        cio.odom_pose(seg, t0 + 5 * 10**9)
