"""extrinsics_from_poles on rendered orange poles with a known camera pose."""

import math

import capture_io as cio
import cv2
import extrinsics_from_poles as efp
import numpy as np
import pytest
import synth_capture as sc

CAM = sc.intrinsics_dict(1280, 720, 818.0, 820.0, 616.0, 393.0, d=(-0.2, 0.05, 0.0, 0.0, 0.0))
TRUE_XYZ = [0.30, 0.01, 0.12]
TRUE_RPY = [sc.OPTICAL_RPY[0] - 0.02, 0.01, sc.OPTICAL_RPY[2] + 0.04]
FLOOR_Z = -0.38
POLE_R, POLE_H = 0.065, 0.45
# pole xy in base, and how far the floor sits below/above the standing floor (a lying robot sees a higher floor)
POLES = {"P1": (1.7, 0.0, 0.0), "P2": (1.7, 0.7, 0.0), "P3": (1.7, -0.6, 0.25), "P4": (2.5, 1.0, 0.0),
         "P5": (2.9, -1.0, 0.25), "P6": (3.2, 0.7, 0.0), "P7": (4.0, 0.2, 0.25)}
LIDAR = np.array([0.289, 0.0])


def _scene(root, name, xy, dz, rng):
    cam = cio.camera_model_from_dict(CAM)
    to_cam = cio.invert_transform(cio.base_from_camera(TRUE_XYZ, TRUE_RPY))
    c = np.array(xy)
    floor_z = FLOOR_Z + dz
    # image: dense volume of the pole drawn as orange on grey
    r = POLE_R * np.sqrt(rng.uniform(0, 1, 9000))
    a = rng.uniform(0, 2 * np.pi, 9000)
    vol = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), floor_z + rng.uniform(0, POLE_H, 9000)], 1)
    u, v, _ = cio.project_points(cio.transform_points(to_cam, vol), cam, 0.1, 20, 3.0)
    img = np.full((720, 1280, 3), 200, np.uint8)
    for x, y in zip(u, v):
        cv2.circle(img, (int(round(x)), int(round(y))), 1, (0, 128, 255), -1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # LiDAR (base frame): floor grid, a far wall, and the pole's LiDAR-facing surface
    gx, gy = np.meshgrid(np.arange(0.5, 5.0, 0.1), np.arange(-2.0, 2.0, 0.1))
    floor = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, floor_z)], 1)
    wall = np.stack([np.full(400, 5.5), rng.uniform(-2, 2, 400), rng.uniform(floor_z, 1.0, 400)], 1)
    toward = np.arctan2(*(LIDAR - c)[::-1])
    th = toward + rng.uniform(-1.2, 1.2, 160)
    pole = np.stack([c[0] + POLE_R * np.cos(th), c[1] + POLE_R * np.sin(th),
                     floor_z + rng.uniform(0.02, POLE_H, 160)], 1)
    pts_base = np.concatenate([floor, wall, pole])
    odom_from_base = sc.pose(1.0, 2.0, 0.5)
    pts_odom = cio.transform_points(odom_from_base, pts_base)
    t0 = 1_000_000_000_000
    meta = {"camera_frame": "front_camera_optical_frame", "latency_s": 0.0, "intrinsics": CAM,
            "extrinsics": sc.extrinsics_dict([0.32, 0, 0.03], sc.OPTICAL_RPY, nominal=True)}
    sc.write_segment(root, f"scene_{name}", meta, [(t0 + 200_000_000, img)],
                     [(t0 + i * 100_000_000, pts_odom, "odom") for i in range(3)],
                     [(t0 + i * 20_000_000, odom_from_base, 0.0) for i in range(60)])


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("poles")
    rng = np.random.default_rng(3)
    for name, (x, y, dz) in POLES.items():
        _scene(root, name, (x, y), dz, rng)
    cio.write_yaml(root / "intr.yaml", CAM)
    return root


def _spec():
    # tape selector only: ahead/left in feet from the nominal lens point on the floor
    return ",".join(f"{n}:{(x - 0.32) / efp.FT:.3f}:{y / efp.FT:.3f}" for n, (x, y, _) in POLES.items())


def test_orange_mask_picks_the_tall_component(capture):
    seg = cio.load_segment(capture, "scene_P1")
    mask, (ub, vb) = efp.orange_mask(cio.read_image(seg.frame_files[0]))
    ys, xs = np.nonzero(mask)
    assert (ys.max() - ys.min()) > 1.5 * (xs.max() - xs.min())
    assert 0 < ub < 1280 and 0 < vb < 720


def test_recovers_known_pose_and_leave_one_out(capture, capsys):
    out = capture / "ext.yaml"
    rc = efp.main(["--capture", str(capture), "--intrinsics", str(capture / "intr.yaml"), "--poles", _spec(),
                   "--out", str(out), "--leave-one-out", "--overlay-dir", str(capture / "ov")])
    assert rc == 0, capsys.readouterr().out
    ext = cio.load_extrinsics(str(out))
    assert ext.nominal is False
    np.testing.assert_allclose(np.degrees(ext.rpy), np.degrees(TRUE_RPY), atol=0.5)
    np.testing.assert_allclose(ext.xyz, TRUE_XYZ, atol=0.03)
    assert (capture / "ov" / "pole_P1.png").exists()


def test_missing_pole_fails_and_writes_nothing(capture):
    out = capture / "ext_bad.yaml"
    far = ",".join(item if not item.startswith("P7:") else "P7:40:0" for item in _spec().split(","))  # no cluster there
    rc = efp.main(["--capture", str(capture), "--intrinsics", str(capture / "intr.yaml"), "--poles", far,
                   "--out", str(out)])
    assert rc == 2 and not out.exists()


def test_rotation_gap_is_zero_for_equal_poses():
    p = np.array([0.3, 0.0, 0.1, *TRUE_RPY])
    assert efp.rotation_gap_deg(p, p) == pytest.approx(0.0, abs=1e-6)
    q = p.copy()
    q[5] += math.radians(2.0)
    assert efp.rotation_gap_deg(p, q) == pytest.approx(2.0, abs=0.05)


def test_evaluate_scores_a_pose_without_fitting(capture, tmp_path):
    good = tmp_path / "good.yaml"
    bad = tmp_path / "bad.yaml"
    cio.write_yaml(good, sc.extrinsics_dict(TRUE_XYZ, TRUE_RPY))
    cio.write_yaml(bad, sc.extrinsics_dict(TRUE_XYZ, [TRUE_RPY[0], TRUE_RPY[1], TRUE_RPY[2] + math.radians(5.0)]))
    args = ["--capture", str(capture), "--intrinsics", str(capture / "intr.yaml"), "--poles", _spec()]
    assert efp.main(args + ["--evaluate", str(good)]) == 0
    assert efp.main(args + ["--evaluate", str(bad)]) == 2
