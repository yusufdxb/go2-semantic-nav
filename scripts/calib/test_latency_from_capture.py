"""latency_from_capture on a synthetic yaw: camera frames stamped a known lag late."""

import math

import cv2
import latency_from_capture as lfc
import numpy as np
import pytest
import synth_capture as sc

W, H, FX = 320, 240, 260.0
CX, CY = (W - 1) / 2, (H - 1) / 2
FPS = 15.0
RES = 0.003  # panorama rad per column


def _panorama(seed=0) -> np.ndarray:
    cols = int(round(2 * math.pi / RES))
    noise = np.random.default_rng(seed).integers(0, 255, (H, cols)).astype(np.uint8)
    pano = cv2.GaussianBlur(noise, (0, 0), 2.0)
    return cv2.normalize(pano, None, 0, 255, cv2.NORM_MINMAX)


def _yaw_capture(tmp_path, lag_s, amplitude=0.3, freq=0.4, duration=8.0, latency_used=0.05):
    pano = _panorama()
    alpha = np.arctan((np.arange(W) - CX) / FX)  # ray azimuth to the right of the optical axis
    map_y = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)
    frames = []
    for i in range(int(duration * FPS)):
        t = i / FPS
        heading = amplitude * math.sin(2 * math.pi * freq * t)
        azimuth = (heading - alpha) % (2 * math.pi)  # world azimuth (CCW) seen by each column
        map_x = np.repeat((azimuth / RES).astype(np.float32)[None, :], H, axis=0)
        img = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        stamp = int(round((t + lag_s) * 1e9)) + 10**12
        frames.append((stamp, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    odom = []
    for j in range(int(duration * 100) + 60):
        t = j / 100.0 - 0.3
        heading = amplitude * math.sin(2 * math.pi * freq * t)
        wz = amplitude * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t)
        odom.append((int(round(t * 1e9)) + 10**12, sc.pose(yaw=heading), wz))
    meta = {"latency_s": latency_used, "intrinsics": sc.intrinsics_dict(W, H, FX, FX, CX, CY),
            "extrinsics": sc.extrinsics_dict([0.3, 0, 0], sc.OPTICAL_RPY)}
    sc.write_segment(tmp_path, "yaw", meta, frames, (), odom)


@pytest.mark.parametrize("lag_s", [0.12, -0.04])
def test_recovers_known_lag(tmp_path, capsys, lag_s):
    _yaw_capture(tmp_path, lag_s)
    assert lfc.main(["--capture", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    lag_ms = float(out.split("lag ")[1].split(" ms")[0])
    assert lag_ms / 1000 == pytest.approx(lag_s, abs=1 / FPS)  # within one frame
    assert abs(lag_ms / 1000 - lag_s) < 0.02  # and in practice much better
    recommended = float(out.split("recommended latency_s ")[1].split(" s")[0])
    assert recommended == pytest.approx(0.05 + lag_s, abs=0.02)


def test_refuses_without_motion(tmp_path):
    _yaw_capture(tmp_path, 0.1, amplitude=0.0)
    assert lfc.main(["--capture", str(tmp_path)]) == 2
