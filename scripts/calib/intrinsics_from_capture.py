#!/usr/bin/env python3
"""Front-camera intrinsics from the ``checkerboard`` segment of a capture.

    python3 scripts/calib/intrinsics_from_capture.py --capture <root> \\
        --board 7x6 --square 0.025 --out front_camera_intrinsics.yaml

Detects the board (inner corners COLSxROWS), keeps a spatially diverse subset
of at most ``--max-views`` views, runs a plumb_bob calibration and writes a ROS
camera_calibration (ost.yaml) file with no ``nominal`` key.

PASS (exit 0) needs: RMS reprojection error <= 0.5 px, at least 15 views, and
corners in at least 75 % of a 4x4 grid over the image. On FAIL (exit 2) the
file is not written unless ``--force``.
"""

from __future__ import annotations

import argparse
import sys

import capture_io as cio
import cv2
import numpy as np

MAX_RMS_PX = 0.5
MIN_VIEWS = 15
MIN_COVERAGE = 0.75
GRID = 4


def parse_board(text: str) -> tuple[int, int]:
    cols, rows = text.lower().split("x")
    return int(cols), int(rows)


def detect_corners(seg: cio.Segment, pattern: tuple[int, int], max_attempts: int) -> list[tuple[int, np.ndarray]]:
    n = len(seg.frame_files)
    idx = np.unique(np.linspace(0, n - 1, min(n, max_attempts)).round().astype(int))
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    found = []
    for i in idx:
        gray = cio.read_image(seg.frame_files[i], gray=True)
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if not ok:
            continue
        c = corners.reshape(-1, 2)
        spacing = float(np.median(np.linalg.norm(np.diff(c.reshape(pattern[1], pattern[0], 2), axis=1), axis=2)))
        win = int(np.clip(spacing / 3.0, 3, 11))
        corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
        found.append((int(i), corners.reshape(-1, 2)))
    return found


def view_features(corners: np.ndarray, pattern: tuple[int, int], width: int, height: int) -> np.ndarray:
    grid = corners.reshape(pattern[1], pattern[0], 2)
    centre = corners.mean(axis=0) / [width, height]
    area = cv2.contourArea(cv2.convexHull(corners.astype(np.float32))) / float(width * height)
    top = np.linalg.norm(grid[0, -1] - grid[0, 0])
    bottom = np.linalg.norm(grid[-1, -1] - grid[-1, 0])
    left = np.linalg.norm(grid[-1, 0] - grid[0, 0])
    right = np.linalg.norm(grid[-1, -1] - grid[0, -1])
    tilt = [np.log(top / bottom), np.log(left / right)]
    return np.array([centre[0], centre[1], np.sqrt(area), *tilt])


def select_diverse(features: np.ndarray, cap: int) -> list[int]:
    """Greedy farthest-point selection in (centre, size, tilt) space."""
    if len(features) <= cap:
        return list(range(len(features)))
    scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    chosen = [int(np.argmax(np.linalg.norm(scaled, axis=1)))]
    dist = np.linalg.norm(scaled - scaled[chosen[0]], axis=1)
    while len(chosen) < cap:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(scaled - scaled[nxt], axis=1))
    return sorted(chosen)


def coverage(corner_sets: list[np.ndarray], width: int, height: int) -> float:
    cells = set()
    for c in corner_sets:
        cx = np.clip((c[:, 0] / width * GRID).astype(int), 0, GRID - 1)
        cy = np.clip((c[:, 1] / height * GRID).astype(int), 0, GRID - 1)
        cells.update(zip(cx.tolist(), cy.tolist()))
    return len(cells) / float(GRID * GRID)


def ost_dict(width: int, height: int, k: np.ndarray, d: np.ndarray, p: np.ndarray, name: str) -> dict:
    return {
        "image_width": int(width),
        "image_height": int(height),
        "camera_name": name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in k.ravel()]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [float(v) for v in d.ravel()[:5]]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        "projection_matrix": {"rows": 3, "cols": 4, "data": [float(v) for v in p.ravel()]},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--segment", default="checkerboard")
    ap.add_argument("--board", default="7x6", help="inner corners COLSxROWS")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres (measure the print)")
    ap.add_argument("--out", default="front_camera_intrinsics.yaml")
    ap.add_argument("--camera-name", default="go2_front_camera")
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--max-attempts", type=int, default=400, help="frames scanned for the board")
    ap.add_argument("--force", action="store_true", help="write the file even when a criterion fails")
    args = ap.parse_args(argv)

    seg = cio.load_segment(args.capture, args.segment)
    width, height = seg.frame_size
    pattern = parse_board(args.board)
    found = detect_corners(seg, pattern, args.max_attempts)
    print(f"[intrinsics] {len(found)} of {len(seg.frame_files)} frames show the {args.board} board ({width}x{height})")
    if len(found) < 3:
        print("[intrinsics] FAIL: fewer than 3 board detections, cannot calibrate")
        return 2

    feats = np.array([view_features(c, pattern, width, height) for _, c in found])
    chosen = [found[i] for i in select_diverse(feats, args.max_views)]
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * args.square
    objs = [obj] * len(chosen)
    imgs = [c.reshape(-1, 1, 2).astype(np.float32) for _, c in chosen]
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (width, height), None, None)

    per_view = []
    for o, im, rv, tv in zip(objs, imgs, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, k, d)
        err = np.linalg.norm(proj.reshape(-1, 2) - im.reshape(-1, 2), axis=1)
        per_view.append((float(np.sqrt(np.mean(err**2))), float(err.max())))
    cov = coverage([c for _, c in chosen], width, height)
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, (width, height), 0)
    p = np.hstack([new_k, np.zeros((3, 1))])

    print(f"[intrinsics] views used {len(chosen)}, RMS {rms:.3f} px, worst view RMS "
          f"{max(v[0] for v in per_view):.3f} px, worst corner {max(v[1] for v in per_view):.3f} px")
    print(f"[intrinsics] fx {k[0, 0]:.2f} fy {k[1, 1]:.2f} cx {k[0, 2]:.2f} cy {k[1, 2]:.2f} "
          f"D {np.array2string(d.ravel()[:5], precision=4)}")
    print(f"[intrinsics] coverage {cov:.2f} of a {GRID}x{GRID} grid")

    fails = []
    if rms > MAX_RMS_PX:
        fails.append(f"RMS {rms:.3f} px > {MAX_RMS_PX}")
    if len(chosen) < MIN_VIEWS:
        fails.append(f"{len(chosen)} views < {MIN_VIEWS}")
    if cov < MIN_COVERAGE:
        fails.append(f"coverage {cov:.2f} < {MIN_COVERAGE} (show the board in the image corners)")

    data = ost_dict(width, height, k, d, p, args.camera_name)
    cio.camera_model_from_dict(data)  # the depth node must be able to load it
    if fails and not args.force:
        print("[intrinsics] FAIL: " + "; ".join(fails) + f". {args.out} NOT written (use --force to override)")
        return 2
    cio.write_yaml(args.out, data, header=f"GO2 front camera intrinsics from {len(chosen)} checkerboard views, "
                                         f"RMS {rms:.3f} px.")
    print(f"[intrinsics] wrote {args.out}")
    if fails:
        print("[intrinsics] FAIL (forced write): " + "; ".join(fails))
        return 2
    print("[intrinsics] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
