"""intrinsics_from_capture on rendered checkerboard views with known K and D."""

import math

import capture_io as cio
import cv2
import intrinsics_from_capture as itc
import numpy as np
import pytest
import synth_capture as sc

W, H = 640, 360
K_TRUE = np.array([[330.0, 0.0, 322.3], [0.0, 331.0, 181.7], [0.0, 0.0, 1.0]])
D_TRUE = np.array([-0.22, 0.06, 0.0008, -0.0005, -0.007])
COLS, ROWS, SQ = 7, 6, 0.025  # inner corners
SS = 4  # supersampling


def _render_board(rvec, tvec) -> np.ndarray | None:
    """White image with the 8x7-square board; None if any of it leaves the frame."""
    ks = K_TRUE.copy()
    ks[0, 0] *= SS
    ks[1, 1] *= SS
    ks[0, 2] = SS * K_TRUE[0, 2] + (SS - 1) / 2
    ks[1, 2] = SS * K_TRUE[1, 2] + (SS - 1) / 2
    outer = np.array([[-SQ, -SQ, 0], [(COLS) * SQ, -SQ, 0], [(COLS) * SQ, ROWS * SQ, 0], [-SQ, ROWS * SQ, 0]])
    o, _ = cv2.projectPoints(outer, rvec, tvec, K_TRUE, D_TRUE)
    if (o[:, 0, 0] < 8).any() or (o[:, 0, 0] > W - 8).any() or (o[:, 0, 1] < 8).any() or (o[:, 0, 1] > H - 8).any():
        return None
    img = np.full((H * SS, W * SS), 255, np.uint8)
    t = np.linspace(0, 1, 6)
    for i in range(COLS + 1):
        for j in range(ROWS + 1):
            if (i + j) % 2:
                continue
            x0, y0 = (i - 1) * SQ, (j - 1) * SQ
            edge = [(x0 + SQ * a, y0) for a in t] + [(x0 + SQ, y0 + SQ * a) for a in t]
            edge += [(x0 + SQ * (1 - a), y0 + SQ) for a in t] + [(x0, y0 + SQ * (1 - a)) for a in t]
            pts = np.array([[x, y, 0.0] for x, y in edge])
            px, _ = cv2.projectPoints(pts, rvec, tvec, ks, D_TRUE)
            cv2.fillPoly(img, [np.round(px.reshape(-1, 2) * 16).astype(np.int32)], 0, lineType=cv2.LINE_AA, shift=4)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def _views(n_wanted: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    centre = np.array([(COLS - 1) * SQ / 2, (ROWS - 1) * SQ / 2, 0.0])
    views, targets = [], [(u, v) for v in np.linspace(0.18, 0.82, 4) for u in np.linspace(0.15, 0.85, 5)]
    for _ in range(400):
        if len(views) >= n_wanted:
            break
        u, v = targets[len(views) % len(targets)]
        z = rng.uniform(0.32, 0.5)
        rvec = np.radians(rng.uniform(-25, 25, 3)) * [1, 1, 0.5]
        r, _ = cv2.Rodrigues(rvec)
        c_cam = np.array([(u * W - K_TRUE[0, 2]) / K_TRUE[0, 0] * z, (v * H - K_TRUE[1, 2]) / K_TRUE[1, 1] * z, z])
        tvec = c_cam - r @ centre
        img = _render_board(rvec, tvec)
        if img is not None:
            views.append(img)
    return views


def _capture(tmp_path, images):
    cam = sc.intrinsics_dict(W, H, 300.0, 300.0, 319.5, 179.5, nominal=True)
    frames = [(1_000_000_000 + i * 66_000_000, cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)) for i, im in enumerate(images)]
    sc.write_segment(tmp_path, "checkerboard", {"latency_s": 0.0, "intrinsics": cam,
                                                "extrinsics": sc.extrinsics_dict([0.3, 0, 0], sc.OPTICAL_RPY)},
                     frames)


@pytest.fixture(scope="module")
def board_views():
    return _views(30)


def test_recovers_known_intrinsics(tmp_path, board_views):
    _capture(tmp_path, board_views)
    out = tmp_path / "intr.yaml"
    rc = itc.main(["--capture", str(tmp_path), "--board", "7x6", "--square", "0.025", "--out", str(out)])
    assert rc == 0
    cam = cio.load_camera_model(str(out))
    assert cam.nominal is False
    assert cam.fx == pytest.approx(K_TRUE[0, 0], rel=0.01)
    assert cam.fy == pytest.approx(K_TRUE[1, 1], rel=0.01)
    assert abs(cam.cx - K_TRUE[0, 2]) < 3 and abs(cam.cy - K_TRUE[1, 2]) < 3
    assert cam.d[0] == pytest.approx(D_TRUE[0], abs=0.03)
    # Same projection as the truth near the image corner (where distortion matters most).
    p = np.array([[0.35, 0.2, 1.0]])
    a, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), cam.k, cam.d)
    b, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K_TRUE, D_TRUE)
    assert np.linalg.norm(a - b) < 1.5
    assert "nominal" not in out.read_text()


def test_too_few_views_fails_and_writes_nothing(tmp_path, board_views):
    _capture(tmp_path, board_views[:8])
    out = tmp_path / "intr.yaml"
    assert itc.main(["--capture", str(tmp_path), "--out", str(out)]) == 2
    assert not out.exists()


def test_parse_board_and_coverage():
    assert itc.parse_board("7x6") == (7, 6)
    corners = np.array([[10.0, 10.0], [630.0, 350.0]])
    assert itc.coverage([corners], W, H) == pytest.approx(2 / 16)
    assert math.isclose(itc.coverage([np.array([[x, y] for x in range(0, W, 40) for y in range(0, H, 40)])], W, H), 1.0)
