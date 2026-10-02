#!/usr/bin/env python3
"""Detector-free geometry acceptance on the ``scene`` segment.

    python3 scripts/calib/acceptance_from_capture.py --capture <root> \\
        --intrinsics front_camera_intrinsics.yaml --extrinsics front_camera_extrinsics.yaml

meta.yaml lists ``taped_objects: [{label, x_m, y_m, z_m (optional, default 0)}]``
in the base frame at capture start. Tape the point of the object's surface
FACING the robot (the LiDAR measures the surface, not the centre). For each
object, its taped point is projected into the image and the median depth of
the sparse depth image lidar_depth_node would publish (same accumulation,
splat, visibility filter and depth limits) in a 40x40 px window around it is
compared with the taped range (camera-frame z).

The range check above is invariant to the camera calibration (the taped point
and the LiDAR returns are both base-frame geometry, projected the same way), so
a second check ties the result to the IMAGE: LiDAR silhouette returns are
projected with the given calibration and their mean distance to image edges
(the extrinsics tool's score) must be <= ``--max-edge-px`` (default 2.0 px).

PASS (exit 0) when every object is in view, has LiDAR returns in its window,
is within 0.15 m, AND the edge alignment passes. Otherwise FAIL (exit 2). The
open-vocab detector is NOT exercised: this checks camera calibration + LiDAR +
TF geometry only.
"""

from __future__ import annotations

import argparse
import sys

import capture_io as cio
import extrinsics_from_capture as exf
import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--segment", default="scene")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--window", type=int, default=40, help="window side in px")
    ap.add_argument("--tolerance", type=float, default=0.15, help="metres")
    ap.add_argument("--max-edge-px", type=float, default=2.0, help="image/LiDAR silhouette alignment limit")
    ap.add_argument("--skip-alignment", action="store_true", help="range check only (does not test the camera)")
    args = ap.parse_args(argv)

    seg = cio.load_segment(args.capture, args.segment)
    cam = cio.load_camera_model(args.intrinsics)
    ext = cio.load_extrinsics(args.extrinsics)
    objects = seg.meta.get("taped_objects") or []
    if not objects:
        print("[acceptance] FAIL: meta.yaml has no taped_objects")
        return 2
    if (cam.width, cam.height) != seg.frame_size:
        print(f"[acceptance] FAIL: intrinsics are {cam.width}x{cam.height}, frames are {seg.frame_size}")
        return 2
    for label, cal in (("intrinsics", cam), ("extrinsics", ext)):
        if cal.nominal:
            print(f"[acceptance] WARN: {label} file is nominal (not measured)")

    start = int(seg.frame_stamps[0])
    odom_from_base_start = cio.odom_pose(seg, start)
    n = len(seg.frame_files)
    idx = np.unique(np.linspace(0, n - 1, min(n, args.frames)).round().astype(int))
    depths = {int(i): cio.node_depth_mm(seg, cam, ext.xyz, ext.rpy, int(seg.frame_stamps[i])) for i in idx}
    half = args.window // 2

    all_ok = True
    print(f"[acceptance] {len(idx)} frames, tolerance {args.tolerance:.2f} m, window {args.window} px")
    for obj in objects:
        label = str(obj.get("label", "object"))
        p_base = np.array([float(obj["x_m"]), float(obj["y_m"]), float(obj.get("z_m", 0.0))])
        p_odom = cio.transform_points(odom_from_base_start, p_base[None, :])
        measured, expected, in_view = [], [], 0
        for i in idx:
            stamp = int(seg.frame_stamps[i])
            pc = cio.transform_points(cio.camera_from_odom(seg, stamp, ext.xyz, ext.rpy), p_odom)
            u, v, z = cio.project_points(pc, cam, cio.NODE_MIN_DEPTH_M, cio.NODE_MAX_DEPTH_M,
                                         cio.NODE_MAX_NORMALIZED_RADIUS)
            if u.size == 0:
                continue
            in_view += 1
            ui, vi = int(round(u[0])), int(round(v[0]))
            win = depths[int(i)][max(0, vi - half): vi + half, max(0, ui - half): ui + half]
            vals = win[win > 0]
            expected.append(float(z[0]))
            if vals.size:
                measured.append(float(np.median(vals)) / 1000.0)
        if in_view == 0:
            print(f"[acceptance] FAIL {label}: taped point is out of view (or outside 0.2-8 m)")
            all_ok = False
            continue
        if not measured:
            print(f"[acceptance] FAIL {label}: no LiDAR returns in the {args.window}x{args.window} window")
            all_ok = False
            continue
        exp, meas = float(np.median(expected)), float(np.median(measured))
        ok = abs(meas - exp) <= args.tolerance
        all_ok &= ok
        print(f"[acceptance] {'ok  ' if ok else 'FAIL'} {label}: taped range {exp:.2f} m, LiDAR depth {meas:.2f} m, "
              f"error {meas - exp:+.2f} m ({len(measured)}/{len(idx)} frames with returns)")

    if args.skip_alignment:
        print("[acceptance] edge alignment SKIPPED: the camera calibration is not tested")
    else:
        frames = exf.prepare_frames(seg, 3, 0.5, 15.0)
        lidar_xyz = seg.meta.get("lidar_xyz", exf.DEFAULT_LIDAR_XYZ)
        for f in frames:
            f.disc_base = exf.discontinuities(f.pts_base, lidar_xyz, 0.3, 0.5)
        n_disc = sum(f.disc_base.shape[0] for f in frames)
        if n_disc < exf.MIN_DISC_POINTS:
            print(f"[acceptance] FAIL edge alignment: only {n_disc} silhouette returns, cannot check the camera")
            all_ok = False
        else:
            edge = exf.score(frames, cam, np.array([*ext.xyz, *ext.rpy]), 15.0)
            ok = edge <= args.max_edge_px
            all_ok &= ok
            print(f"[acceptance] {'ok  ' if ok else 'FAIL'} edge alignment: {edge:.2f} px mean distance of "
                  f"{n_disc} LiDAR silhouette returns to image edges (limit {args.max_edge_px:.1f})")

    print(f"[acceptance] {'PASS' if all_ok else 'FAIL'} (geometry only: the detector is not exercised)")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
