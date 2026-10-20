#!/usr/bin/env python3
"""Fit the camera pose in the LiDAR base frame from single-pole captures.

    python3 scripts/calib/extrinsics_from_poles.py --capture <root> \\
        --intrinsics front_camera_intrinsics.yaml --poles C1:4:0,C2:4:2,... \\
        --out front_camera_extrinsics.yaml [--leave-one-out] [--overlay-dir dir]

``--poles`` lists ``<scene>:<ahead_ft>:<left_ft>`` where the scene folder is
``<capture>/scene_<scene>``. The tape position only SELECTS which LiDAR cluster
is the pole (a 0.6 m window); it does not enter the objective.

Each capture: one orange pole, robot still. Pole pixels come from an orange HSV
mask; pole LiDAR points are the cluster at the pole. Objective (lower is
better): mean distance (px, capped) from each projected pole point to the mask,
plus the squared (scaled by a 2 cm centroid noise) metric error of the mask's bottom-centre against the
projected point where the pole's near surface meets the RANSAC floor plane (the
LiDAR sees that surface, and so does the camera: no radius correction).

The pose is fitted in the LiDAR's own base frame, so it absorbs any LiDAR mount
error: use the result with the raw ``/go2/lidar/points`` cloud, as
``lidar_depth_node`` does.

``--evaluate <extrinsics.yaml>`` fits nothing: it scores that pose on the listed captures (use captures that
were not in any fit) and exits 0 only when every pole has a mean point-to-mask distance <= 2 px and a
bottom-point error <= 4 cm.

Exit 0 when the fit finished and every restart agrees (0.5 deg, 1 cm); exit 2
when restarts disagree or a capture has no usable pole (output not written
unless --force).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import capture_io as cio
import cv2
import numpy as np
from extrinsics_from_capture import _bilinear, nelder_mead
from scipy import ndimage

FT = 0.3048
SELECT_WINDOW_M = 0.6
HEIGHT_BAND_M = (0.10, 0.60)
VOXEL_M = 0.03
MAX_POLE_WIDTH_M = 0.35
DIST_CAP_PX = 60.0
BOTTOM_SIGMA_M = 0.02  # LiDAR cluster centroid noise at the pole, metres
BOTTOM_WEIGHT = 0.25
BOTTOM_CAP_M = 0.15
LOO_BOTTOM_M = 0.04
EVAL_POINT_PX = 2.0
EVAL_BOTTOM_M = 0.04
NOMINAL_CAM_XYZ = np.array([0.32, 0.0, 0.03])
RESTART_AGREE_DEG = 0.5
RESTART_AGREE_M = 0.01
# Scenes whose 0.6 m selector window also holds another object: seed the cluster here (base frame xy).
SEEDS: dict[str, tuple[float, float]] = {"C5": (2.604, -0.856), "C7": (3.227, -0.471)}


@dataclass
class PoleData:
    name: str
    dist: np.ndarray  # px distance to the orange mask (0 inside), capped
    bottom_px: np.ndarray  # (u, v) of the mask's bottom centre
    pts: np.ndarray  # Nx3 pole returns, base frame
    floor_pt: np.ndarray  # 3, pole near surface at the floor, base frame
    n_mask_px: int


def orange_mask(bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Largest tall orange component, and its bottom-centre (u, v)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 110, 55), (18, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best, best_area = -1, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h >= 1.5 * w and area > best_area:
            best, best_area = i, area
    if best < 0:
        raise ValueError("no tall orange component found")
    mask = (lab == best).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    vb = float(ys.max())
    ub = float(xs[ys >= vb - 4].mean())
    return mask, (ub, vb)


def floor_height(z: np.ndarray) -> float:
    """Floor height in the (gravity-aligned) odom frame: the fullest 2 cm bin among the lowest 0.5 m of returns."""
    z = z[z < z.min() + 0.5]
    bins = np.arange(z.min(), z.max() + 0.02, 0.02)
    hist, edges = np.histogram(z, bins)
    i = int(np.argmax(hist))
    in_bin = z[(z >= edges[i] - 0.01) & (z < edges[i + 1] + 0.01)]
    return float(np.median(in_bin))


def select_pole(pts: np.ndarray, z0: float, seed_xy: np.ndarray) -> np.ndarray:
    """Pole returns (odom frame): the connected cluster (3 cm voxels, 8-neighbour) of returns 0.10-0.60 m above
    the floor, within SELECT_WINDOW_M of ``seed_xy``, that is at most MAX_POLE_WIDTH_M wide, has at least 20
    returns, and lies nearest the seed."""
    h = pts[:, 2] - z0
    near = (h > HEIGHT_BAND_M[0]) & (h < HEIGHT_BAND_M[1]) & (np.linalg.norm(pts[:, :2] - seed_xy, axis=1) < SELECT_WINDOW_M)
    cand = pts[near]
    if len(cand) < 20:
        raise ValueError("too few LiDAR returns near the tape position")
    g = np.floor(cand[:, :2] / VOXEL_M).astype(int)
    g -= g.min(0)
    grid = np.zeros(g.max(0) + 1, bool)
    grid[g[:, 0], g[:, 1]] = True
    lab, k = ndimage.label(grid, structure=np.ones((3, 3)))
    labels = lab[g[:, 0], g[:, 1]]
    best, best_d = None, np.inf
    for j in range(1, k + 1):
        q = cand[labels == j]
        if len(q) < 20 or (np.ptp(q[:, :2], axis=0) > MAX_POLE_WIDTH_M).any():
            continue
        d = float(np.linalg.norm(np.median(q[:, :2], axis=0) - seed_xy))
        if d < best_d:
            best, best_d = q, d
    if best is None:
        raise ValueError("no compact LiDAR cluster near the tape position")
    return best


def tape_seed_odom(odom_from_base: np.ndarray, z0: float, ahead_ft: float, left_ft: float) -> np.ndarray:
    """Tape position in odom xy: from the nominal lens point on the floor, along the heading and to the left."""
    r = odom_from_base[:3, :3]
    fwd = r[:, 0] * [1, 1, 0]
    fwd /= np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0], 0.0])
    lens = cio.transform_points(odom_from_base, NOMINAL_CAM_XYZ[None, :])[0]
    return (lens + ahead_ft * FT * fwd + left_ft * FT * left)[:2]


def load_pole(capture: str, name: str, ahead_ft: float, left_ft: float) -> PoleData:
    seg = cio.load_segment(capture, f"scene_{name}")
    mid = len(seg.frame_files) // 2
    stamp = int(seg.frame_stamps[mid])
    bgr = cio.read_image(seg.frame_files[mid])
    mask, bottom = orange_mask(bgr)
    pts_odom = cio.clouds_between(seg, int(seg.cloud_stamps[0]), int(seg.cloud_stamps[-1]))
    odom_from_base = cio.odom_pose(seg, stamp)
    z0 = floor_height(pts_odom[:, 2])
    if name in SEEDS:
        seed = cio.transform_points(odom_from_base, np.array([[*SEEDS[name], z0]]))[0, :2]
    else:
        seed = tape_seed_odom(odom_from_base, z0, ahead_ft, left_ft)
    pole = select_pole(pts_odom, z0, seed)
    centre = np.median(pole[:, :2], axis=0)
    base_from_odom = cio.invert_transform(odom_from_base)
    floor_pt = cio.transform_points(base_from_odom, np.array([[centre[0], centre[1], z0]]))[0]
    dist = np.minimum(cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5), DIST_CAP_PX)
    return PoleData(name, dist, np.array(bottom), cio.transform_points(base_from_odom, pole), floor_pt, int(mask.sum()))


def _project(pts_base: np.ndarray, cam, params: np.ndarray):
    pc = cio.transform_points(cio.invert_transform(cio.base_from_camera(params[:3], params[3:])), pts_base)
    return cio.project_points(pc, cam, cio.NODE_MIN_DEPTH_M, cio.NODE_MAX_DEPTH_M, cio.NODE_MAX_NORMALIZED_RADIUS)


def _lookup(dist: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _bilinear(dist, u, v)


def pole_terms(p: PoleData, cam, params: np.ndarray) -> tuple[float, float, float]:
    """(mean point-to-mask px with misses at the cap, bottom-point error px, bottom-point error m at its depth)."""
    u, v, _ = _project(p.pts, cam, params)
    mean_d = (float(_lookup(p.dist, u, v).sum()) + DIST_CAP_PX * (len(p.pts) - u.size)) / len(p.pts)
    ub, vb, zb = _project(p.floor_pt[None, :], cam, params)
    if not ub.size:
        return mean_d, 4 * DIST_CAP_PX, BOTTOM_CAP_M
    px = float(np.hypot(ub[0] - p.bottom_px[0], vb[0] - p.bottom_px[1]))
    return mean_d, px, min(px * float(zb[0]) / cam.fx, BOTTOM_CAP_M)


def objective(poles: list[PoleData], cam, params: np.ndarray) -> float:
    total = 0.0
    for p in poles:
        d, _, b_m = pole_terms(p, cam, params)
        total += d + BOTTOM_WEIGHT * (b_m / BOTTOM_SIGMA_M) ** 2
    return total / len(poles)


def fit(poles: list[PoleData], cam, starts: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    scale = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
    def run(x0):
        f = lambda q: objective(poles, cam, q)  # noqa: E731
        x = nelder_mead(f, x0, scale, iters=1500)
        return nelder_mead(f, x, scale * 0.25, iters=1500)  # restart: Nelder-Mead stalls in flat valleys

    sols = [run(s) for s in starts]
    vals = [objective(poles, cam, s) for s in sols]
    return sols[int(np.argmin(vals))], sols


def perturbed_starts(p0: np.ndarray, n: int = 5, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = [p0.copy()]
    for _ in range(n - 1):
        out.append(p0 + np.concatenate([rng.uniform(-0.03, 0.03, 3), np.radians(rng.uniform(-3, 3, 3))]))
    return out


def rotation_gap_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = cio.rpy_to_matrix(*a[3:]), cio.rpy_to_matrix(*b[3:])
    return float(np.degrees(np.arccos(np.clip((np.trace(ra.T @ rb) - 1) / 2, -1, 1))))


def parse_poles(text: str) -> list[tuple[str, float, float]]:
    out = []
    for item in text.split(","):
        name, a, left = item.split(":")
        out.append((name, float(a), float(left)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--poles", required=True)
    ap.add_argument("--init-extrinsics", default="", help="default: nominal (xyz 0.32 0 0.03, level camera)")
    ap.add_argument("--out", default="front_camera_extrinsics.yaml")
    ap.add_argument("--evaluate", default="", help="score this extrinsics file on the poles instead of fitting")
    ap.add_argument("--leave-one-out", action="store_true")
    ap.add_argument("--overlay-dir", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cam = cio.load_camera_model(args.intrinsics)
    poles = []
    for name, a, left in parse_poles(args.poles):
        try:
            poles.append(load_pole(args.capture, name, a, left))
        except ValueError as e:
            print(f"[poles] FAIL {name}: {e}")
            return 2
        p = poles[-1]
        print(f"[poles] {name}: mask {p.n_mask_px} px, {len(p.pts)} LiDAR returns, floor point "
              f"({p.floor_pt[0]:.3f}, {p.floor_pt[1]:+.3f}, {p.floor_pt[2]:+.3f})")

    if args.evaluate:
        ext = cio.load_extrinsics(args.evaluate)
        params = np.array([*ext.xyz, *ext.rpy])
        ok = True
        for p in poles:
            d, b, bm = pole_terms(p, cam, params)
            good = d <= EVAL_POINT_PX and bm <= EVAL_BOTTOM_M
            ok &= good
            print(f"[poles] {'ok  ' if good else 'FAIL'} {p.name}: point-to-mask {d:.2f} px, "
                  f"bottom error {b:.2f} px ({bm * 100:.1f} cm)")
        print(f"[poles] evaluate {'PASS' if ok else 'FAIL'} (<= {EVAL_POINT_PX} px, <= {EVAL_BOTTOM_M * 100:.0f} cm)")
        return 0 if ok else 2

    if args.init_extrinsics:
        init = cio.load_extrinsics(args.init_extrinsics)
        p0 = np.array([*init.xyz, *init.rpy])
    else:
        p0 = np.array([*NOMINAL_CAM_XYZ, -np.pi / 2, 0.0, -np.pi / 2])
    starts = perturbed_starts(p0)
    best, sols = fit(poles, cam, starts)
    gaps_deg = [rotation_gap_deg(best, s) for s in sols]
    gaps_m = [float(np.linalg.norm(best[:3] - s[:3])) for s in sols]
    stable = max(gaps_deg) <= RESTART_AGREE_DEG and max(gaps_m) <= RESTART_AGREE_M
    print(f"[poles] objective {objective(poles, cam, p0):.2f} (start) -> {objective(poles, cam, best):.2f}")
    print(f"[poles] xyz {np.round(best[:3], 4).tolist()} rpy {np.round(best[3:], 5).tolist()}")
    print(f"[poles] restart spread: max {max(gaps_deg):.2f} deg, {max(gaps_m) * 100:.2f} cm "
          f"({'stable' if stable else 'UNSTABLE'})")
    for p in poles:
        d, b, bm = pole_terms(p, cam, best)
        print(f"[poles]   {p.name}: point-to-mask {d:.2f} px, bottom error {b:.2f} px ({bm * 100:.1f} cm)")

    loo_ok = True
    if args.leave_one_out:
        for i, p in enumerate(poles):
            rest = poles[:i] + poles[i + 1:]
            sol, _ = fit(rest, cam, [best])
            d, b, bm = pole_terms(p, cam, sol)
            print(f"[poles]   LOO {p.name}: point-to-mask {d:.2f} px, bottom error {b:.2f} px ({bm * 100:.1f} cm)")
            loo_ok &= d <= 4.0 and bm <= LOO_BOTTOM_M

    if args.overlay_dir:
        out_dir = Path(args.overlay_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in poles:
            img = np.zeros((*p.dist.shape, 3), np.uint8)
            img[p.dist == 0] = (0, 140, 255)
            u, v, _ = _project(p.pts, cam, best)
            for x, y in zip(u, v):
                cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)
            ub, vb, _ = _project(p.floor_pt[None, :], cam, best)
            if ub.size:
                cv2.circle(img, (int(ub[0]), int(vb[0])), 6, (0, 255, 0), 2)
            cv2.circle(img, (int(p.bottom_px[0]), int(p.bottom_px[1])), 6, (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / f"pole_{p.name}.png"), img)

    if not stable and not args.force:
        print("[poles] FAIL: restarts disagree, output not written (use --force)")
        return 2
    cio.write_yaml(args.out, {
        "nominal": False, "parent_frame": "base_link", "child_frame": "front_camera_optical_frame",
        "xyz": [round(float(x), 5) for x in best[:3]], "rpy": [round(float(x), 6) for x in best[3:]],
    }, header=f"Camera optical frame in the LiDAR's own base frame, fitted from {len(poles)} pole captures "
              f"(extrinsics_from_poles.py).\nUse with the raw /go2/lidar/points cloud.")
    print(f"[poles] wrote {args.out}" + ("" if loo_ok else " (leave-one-out bar NOT met)"))
    return 0 if (stable and loo_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
