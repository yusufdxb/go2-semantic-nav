#!/usr/bin/env python3
"""Camera stamp latency from the ``yaw`` segment (robot yawing in place).

    python3 scripts/calib/latency_from_capture.py --capture <root>

The image-derived yaw rate (median change in feature azimuth between
consecutive frames, after undistortion) is cross-correlated with the odometry
yaw rate. A positive lag means image stamps are LATER than the true capture
time, so ``latency_s`` on go2_front_camera should grow by that amount:
recommended latency_s = latency_s used at capture + lag.

Refuses (exit 2) when the odom yaw-rate RMS is under 0.2 rad/s (not enough
motion), the peak normalized correlation is under 0.6, or the best lag sits on
the edge of the search range.
"""

from __future__ import annotations

import argparse
import sys

import capture_io as cio
import cv2
import numpy as np

MIN_RATE_RMS = 0.2
MIN_CORR = 0.6
MIN_TRACKS = 20


def image_yaw_rates(seg: cio.Segment, cam: cio.CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """(mid-stamp seconds, yaw rate rad/s) per consecutive frame pair; CCW robot yaw is positive."""
    k, d = cam.k, cam.d
    stamps, rates = [], []
    prev = cio.read_image(seg.frame_files[0], gray=True)
    for i in range(1, len(seg.frame_files)):
        cur = cio.read_image(seg.frame_files[i], gray=True)
        dt = (seg.frame_stamps[i] - seg.frame_stamps[i - 1]) / 1e9
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=7)
        if pts is not None and dt > 0:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, winSize=(21, 21), maxLevel=3)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < 1.0)
            if ok.sum() >= MIN_TRACKS:
                n0 = cv2.undistortPoints(pts[ok].reshape(-1, 1, 2), k, d).reshape(-1, 2)
                n1 = cv2.undistortPoints(nxt[ok].reshape(-1, 1, 2), k, d).reshape(-1, 2)
                # Rotation about the camera's vertical axis changes azimuth atan(x) for every row.
                dazim = np.arctan(n1[:, 0]) - np.arctan(n0[:, 0])
                stamps.append((seg.frame_stamps[i] + seg.frame_stamps[i - 1]) / 2e9)
                rates.append(float(np.median(dazim)) / dt)
        prev = cur
    return np.array(stamps), np.array(rates)


def best_lag(t_img, r_img, t_odom, r_odom, max_lag, step) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Lag L maximizing corr(image(t), odom(t - L)); returns (lag, corr, lags, corrs)."""
    t0, t1 = t_img[0], t_img[-1]
    grid = np.arange(t0, t1, step)
    img = np.interp(grid, t_img, r_img)
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    corrs = np.full(lags.shape, np.nan)
    for j, lag in enumerate(lags):
        src = grid - lag
        valid = (src >= t_odom[0]) & (src <= t_odom[-1])
        if valid.sum() < 0.5 * grid.size:
            continue
        a = img[valid]
        b = np.interp(src[valid], t_odom, r_odom)
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        corrs[j] = float(np.corrcoef(a, b)[0, 1])
    j = int(np.nanargmax(corrs))
    lag = float(lags[j])
    if 0 < j < len(lags) - 1 and np.isfinite(corrs[j - 1]) and np.isfinite(corrs[j + 1]):
        y0, y1, y2 = corrs[j - 1], corrs[j], corrs[j + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)) * step
    return lag, float(corrs[j]), lags, corrs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--segment", default="yaw")
    ap.add_argument("--intrinsics", default="", help="calibrated intrinsics yaml (default: meta.yaml intrinsics)")
    ap.add_argument("--max-lag", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=0.005)
    args = ap.parse_args(argv)

    seg = cio.load_segment(args.capture, args.segment)
    cam = cio.load_camera_model(args.intrinsics) if args.intrinsics else seg.camera_model()
    if seg.odom_stamps.size < 10:
        print("[latency] FAIL: odom.csv has too few samples")
        return 2
    t_odom = seg.odom_stamps / 1e9
    r_odom = seg.odom_values[:, 7]
    rate_rms = float(np.sqrt(np.mean(r_odom**2)))
    t_img, r_img = image_yaw_rates(seg, cam)
    print(f"[latency] {len(seg.frame_files)} frames, {t_img.size} rate samples, odom yaw-rate RMS {rate_rms:.3f} rad/s")
    if rate_rms < MIN_RATE_RMS:
        print(f"[latency] FAIL: yaw-rate RMS {rate_rms:.3f} < {MIN_RATE_RMS} rad/s; yaw the robot faster/longer")
        return 2
    if t_img.size < 10:
        print("[latency] FAIL: too few frame pairs with trackable features")
        return 2

    lag, corr, lags, _ = best_lag(t_img, r_img, t_odom, r_odom, args.max_lag, args.step)
    recommended = seg.latency_s + lag
    print(f"[latency] lag {lag * 1000:+.1f} ms (positive: image stamps late), peak correlation {corr:.3f}")
    print(f"[latency] latency_s used at capture {seg.latency_s:.3f} s -> recommended latency_s {recommended:.3f} s")
    if corr < MIN_CORR:
        print(f"[latency] FAIL: peak correlation {corr:.3f} < {MIN_CORR}; the estimate is not trustworthy")
        return 2
    if abs(lag) >= args.max_lag - args.step:
        print(f"[latency] FAIL: best lag is at the search edge (+/-{args.max_lag} s)")
        return 2
    if recommended < 0:
        print("[latency] WARN: recommended latency_s is negative (camera stamps early); check clocks")
    print("[latency] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
