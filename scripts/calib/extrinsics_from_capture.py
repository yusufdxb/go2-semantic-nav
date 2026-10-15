#!/usr/bin/env python3
"""Refine the camera-to-base extrinsic from the ``scene`` segment (robot static).

    python3 scripts/calib/extrinsics_from_capture.py --capture <root> \\
        --intrinsics front_camera_intrinsics.yaml [--init-extrinsics tape.yaml] \\
        --out front_camera_extrinsics.yaml [--overlay-dir overlays/]

Score: LiDAR silhouette returns are projected into the image; the score is the
mean distance (px, truncated at ``--dist-cap``) from each to the nearest Canny
edge. Lower is better. Silhouettes are found in the LiDAR's OWN view (an
azimuth/elevation range image around the LiDAR origin, meta ``lidar_xyz``,
default the base stack's mount): a return is a silhouette return when a
neighbouring direction (``--disc-bin-deg`` bins, +/-1) ranges more than
``--jump-m`` farther. Doing this in the camera view instead flags false
silhouettes, because the LiDAR sits below the camera and sees background the
camera cannot (parallax of several px at 1 to 5 m).

Search, all around the initial extrinsic (tape measure + nominal rotation):
rpy +/-5 deg (1 deg, then 0.2 deg), xyz +/-5 cm (1 cm, then 0.2 cm), then a
joint 6-D Nelder-Mead refinement (rotation and lateral translation are coupled:
a yaw error and a sideways shift look alike except through parallax).

Exit codes:
  0  PASS: the optimum is inside the search box, well defined, and at least
     0.25 px better than the initial extrinsic. Output written (nominal: false).
  3  NO SIGNIFICANT IMPROVEMENT: well defined and inside the box, but within
     0.25 px of the initial score; the initial values are already as good as
     this scene can show. Output still written.
  2  FAIL, output not written (unless --force): optimum on the search
     boundary (initial guess too far off, re-measure), a rotation axis with
     less than 0.5 px of score contrast at +/-2 deg (scene has too few edges
     in that direction), or fewer than 50 silhouette returns.
A translation axis with less than 0.1 px contrast at +/-2 cm is not
observable from the scene: it keeps the initial (tape) value and is reported.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import capture_io as cio
import cv2
import numpy as np

DEFAULT_LIDAR_XYZ = (0.28945, 0.0, -0.046825)  # base stack go2_localization lidar_xyz default
SEARCH_DEG = 5.0
SEARCH_M = 0.05
MIN_IMPROVEMENT_PX = 0.25
MIN_ROT_CONTRAST_PX = 0.5
MIN_TRANS_CONTRAST_PX = 0.1
MIN_DISC_POINTS = 50


@dataclass
class FrameData:
    stamp_ns: int
    bgr: np.ndarray
    dist: np.ndarray  # distance (px) to nearest image edge, truncated
    pts_base: np.ndarray  # accumulated cloud in the base frame at this frame's stamp
    disc_base: np.ndarray = None  # discontinuity returns in the base frame


def prepare_frames(seg, n_frames, window_s, dist_cap) -> list[FrameData]:
    n = len(seg.frame_files)
    frames = []
    for i in np.unique(np.linspace(0, n - 1, min(n, n_frames)).round().astype(int)):
        stamp = int(seg.frame_stamps[i])
        bgr = cio.read_image(seg.frame_files[i])
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
        pts = cio.clouds_between(seg, stamp - int(window_s * 1e9), stamp + int(window_s * 1e9))
        if pts.shape[0] == 0:
            continue
        base_from_odom = cio.invert_transform(cio.odom_pose(seg, stamp))
        frames.append(FrameData(stamp, bgr, np.minimum(dist, dist_cap), cio.transform_points(base_from_odom, pts)))
    return frames


def _project(pts_base, cam, xyz, rpy):
    pc = cio.transform_points(cio.invert_transform(cio.base_from_camera(xyz, rpy)), pts_base)
    return cio.project_points(pc, cam, cio.NODE_MIN_DEPTH_M, cio.NODE_MAX_DEPTH_M, cio.NODE_MAX_NORMALIZED_RADIUS)


def discontinuities(pts_base: np.ndarray, lidar_xyz, jump_m: float, bin_deg: float) -> np.ndarray:
    """Silhouette returns (base frame): a neighbouring LiDAR direction ranges > jump_m farther."""
    import cv2 as _cv2

    if pts_base.shape[0] == 0:
        return np.empty((0, 3))
    rel = pts_base - np.asarray(lidar_xyz, np.float64)
    rng = np.linalg.norm(rel, axis=1)
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    el = np.degrees(np.arcsin(np.clip(rel[:, 2] / np.maximum(rng, 1e-9), -1, 1)))
    ai = np.floor((az - az.min()) / bin_deg).astype(int)
    ei = np.floor((el - el.min()) / bin_deg).astype(int)
    grid_min = np.full((ei.max() + 1, ai.max() + 1), np.inf, np.float32)
    np.minimum.at(grid_min, (ei, ai), rng.astype(np.float32))
    filled = np.where(np.isfinite(grid_min), grid_min, 0.0).astype(np.float32)
    farthest = _cv2.dilate(filled, np.ones((3, 3), np.uint8))  # empty bins (0) never win a max
    sel = farthest[ei, ai] - rng > jump_m
    return pts_base[sel]


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def score(frames: list[FrameData], cam, params: np.ndarray, dist_cap: float) -> float:
    xyz, rpy = params[:3], params[3:]
    total, count = 0.0, 0
    for f in frames:
        if f.disc_base is None or f.disc_base.shape[0] == 0:
            continue
        u, v, _ = _project(f.disc_base, cam, xyz, rpy)
        total += float(_bilinear(f.dist, u, v).sum()) + dist_cap * (f.disc_base.shape[0] - u.size)
        count += f.disc_base.shape[0]
    return total / count if count else float("inf")


def grid_refine(fn, centre: np.ndarray, axes: list[int], half: float, step: float) -> np.ndarray:
    offsets = np.arange(-half, half + step / 2, step)
    best, best_s = centre.copy(), fn(centre)
    for combo in itertools.product(offsets, repeat=len(axes)):
        p = centre.copy()
        p[axes] += combo
        s = fn(p)
        if s < best_s:
            best, best_s = p, s
    return best


def nelder_mead(fn, x0: np.ndarray, scale: np.ndarray, iters: int = 400, tol: float = 1e-4) -> np.ndarray:
    """Minimal Nelder-Mead in scaled coordinates (x = x0 + scale * y)."""
    n = x0.size
    simplex = [np.zeros(n)] + [np.eye(n)[i] for i in range(n)]
    vals = [fn(x0 + scale * y) for y in simplex]
    for _ in range(iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[-1] - vals[0] < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        refl = centroid + (centroid - simplex[-1])
        f_r = fn(x0 + scale * refl)
        if f_r < vals[0]:
            exp = centroid + 2.0 * (centroid - simplex[-1])
            f_e = fn(x0 + scale * exp)
            simplex[-1], vals[-1] = (exp, f_e) if f_e < f_r else (refl, f_r)
        elif f_r < vals[-2]:
            simplex[-1], vals[-1] = refl, f_r
        else:
            con = centroid + 0.5 * (simplex[-1] - centroid)
            f_c = fn(x0 + scale * con)
            if f_c < vals[-1]:
                simplex[-1], vals[-1] = con, f_c
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (y - simplex[0]) for y in simplex[1:]]
                vals = [vals[0]] + [fn(x0 + scale * y) for y in simplex[1:]]
    return x0 + scale * simplex[int(np.argmin(vals))]


def contrast(fn, p: np.ndarray, axis: int, delta: float) -> float:
    lo, hi = p.copy(), p.copy()
    lo[axis] -= delta
    hi[axis] += delta
    return 0.5 * (fn(lo) + fn(hi)) - fn(p)


def overlay(f: FrameData, cam, xyz, rpy) -> np.ndarray:
    img = f.bgr.copy()
    u, v, z = _project(f.pts_base, cam, xyz, rpy)
    t = np.clip((z - 0.2) / 7.8, 0, 1)
    for x, y, c in zip(np.rint(u).astype(int), np.rint(v).astype(int), t):
        cv2.circle(img, (int(x), int(y)), 1, (int(255 * c), 64, int(255 * (1 - c))), -1)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--segment", default="scene")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--init-extrinsics", default="", help="default: the extrinsics in meta.yaml")
    ap.add_argument("--out", default="front_camera_extrinsics.yaml")
    ap.add_argument("--overlay-dir", default="")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--window-s", type=float, default=0.5, help="clouds within +/- this of each frame")
    ap.add_argument("--jump-m", type=float, default=0.3)
    ap.add_argument("--disc-bin-deg", type=float, default=0.5, help="LiDAR range-image bin for silhouettes")
    ap.add_argument("--dist-cap", type=float, default=15.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seg = cio.load_segment(args.capture, args.segment)
    cam = cio.load_camera_model(args.intrinsics)
    if (cam.width, cam.height) != seg.frame_size:
        print(f"[extrinsics] FAIL: intrinsics are {cam.width}x{cam.height}, frames are {seg.frame_size}")
        return 2
    init = cio.load_extrinsics(args.init_extrinsics) if args.init_extrinsics else seg.extrinsics()
    p0 = np.array([*init.xyz, *init.rpy], dtype=np.float64)

    frames = prepare_frames(seg, args.frames, args.window_s, args.dist_cap)
    lidar_xyz = seg.meta.get("lidar_xyz", DEFAULT_LIDAR_XYZ)
    for f in frames:
        f.disc_base = discontinuities(f.pts_base, lidar_xyz, args.jump_m, args.disc_bin_deg)
    n_disc = sum(f.disc_base.shape[0] for f in frames)
    print(f"[extrinsics] {len(frames)} frames, {n_disc} silhouette returns")
    if n_disc < MIN_DISC_POINTS:
        print(f"[extrinsics] FAIL: fewer than {MIN_DISC_POINTS} silhouette returns; face objects with clear "
              "silhouettes 1 to 3 m away")
        return 2

    def fn(p):
        return score(frames, cam, p, args.dist_cap)

    s_init = fn(p0)
    rot, trans = [3, 4, 5], [0, 1, 2]
    deg = math.radians(1.0)
    p = grid_refine(fn, p0, rot, SEARCH_DEG * deg, 1.0 * deg)
    p = grid_refine(fn, p, rot, 1.0 * deg, 0.2 * deg)
    p = grid_refine(fn, p, trans, SEARCH_M, 0.01)
    p = grid_refine(fn, p, trans, 0.01, 0.002)
    p = nelder_mead(fn, p, np.array([0.01, 0.01, 0.01, 0.5 * deg, 0.5 * deg, 0.5 * deg]))

    kept = []
    for a, name in zip(trans, "xyz"):
        if contrast(fn, p, a, 0.02) < MIN_TRANS_CONTRAST_PX:
            p[a] = p0[a]
            kept.append(name)
    rot_contrast = [contrast(fn, p, a, 2.0 * deg) for a in rot]
    s_best = fn(p)
    off = p - p0
    print(f"[extrinsics] score {s_init:.3f} -> {s_best:.3f} px (lower is better)")
    print(f"[extrinsics] d_rpy deg {np.array2string(np.degrees(off[3:]), precision=2)}, "
          f"d_xyz cm {np.array2string(off[:3] * 100, precision=2)}")
    print(f"[extrinsics] rotation contrast at +/-2 deg (px): {np.array2string(np.array(rot_contrast), precision=3)}")
    if kept:
        print(f"[extrinsics] translation axis {','.join(kept)} not observable here: kept the initial value")

    fails = []
    if np.any(np.abs(np.degrees(off[3:])) >= SEARCH_DEG - 0.1) or np.any(np.abs(off[:3]) >= SEARCH_M - 0.001):
        fails.append("optimum on the search boundary (initial extrinsic too far off; re-measure)")
    weak = [n for n, c in zip(("roll", "pitch", "yaw"), rot_contrast) if c < MIN_ROT_CONTRAST_PX]
    if weak:
        fails.append(f"rotation not well defined for {','.join(weak)} (scene lacks edges)")

    if args.overlay_dir and frames:
        out = Path(args.overlay_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "overlay_before.png"), overlay(frames[0], cam, p0[:3], p0[3:]))
        cv2.imwrite(str(out / "overlay_after.png"), overlay(frames[0], cam, p[:3], p[3:]))
        print(f"[extrinsics] overlays in {out}")

    data = {
        "nominal": False,
        "parent_frame": init.parent_frame,
        "child_frame": init.child_frame,
        "xyz": [round(float(v), 5) for v in p[:3]],
        "rpy": [round(float(v), 6) for v in p[3:]],
    }
    cio.extrinsics_from_dict(data)
    if fails and not args.force:
        print("[extrinsics] FAIL: " + "; ".join(fails) + f". {args.out} NOT written (use --force to override)")
        return 2
    cio.write_yaml(args.out, data, header=f"GO2 front camera optical frame in {init.parent_frame}, refined against "
                                         f"LiDAR silhouettes: score {s_init:.3f} -> {s_best:.3f} px.")
    print(f"[extrinsics] wrote {args.out}")
    if fails:
        print("[extrinsics] FAIL (forced write): " + "; ".join(fails))
        return 2
    if s_init - s_best < MIN_IMPROVEMENT_PX:
        print(f"[extrinsics] NO SIGNIFICANT IMPROVEMENT (< {MIN_IMPROVEMENT_PX} px): initial extrinsic already "
              "optimal for this scene")
        return 3
    print("[extrinsics] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
