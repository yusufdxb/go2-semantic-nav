"""Synthetic capture folders for the calibration-tool tests (no robot, no ROS)."""

from __future__ import annotations

import math
from pathlib import Path

import capture_io as cio
import cv2
import numpy as np


def quaternion_from_matrix(r: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0, r[1, 0] - r[0, 1])
    return x, y, z, w


def intrinsics_dict(width, height, fx, fy, cx, cy, d=(0.0, 0.0, 0.0, 0.0, 0.0), nominal=False) -> dict:
    return {
        "nominal": nominal,
        "image_width": width,
        "image_height": height,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [float(v) for v in d]},
    }


OPTICAL_RPY = [-math.pi / 2, 0.0, -math.pi / 2]


def extrinsics_dict(xyz, rpy, nominal=False) -> dict:
    return {"nominal": nominal, "parent_frame": "base_link", "child_frame": "front_camera_optical_frame",
            "xyz": [float(v) for v in xyz], "rpy": [float(v) for v in rpy]}


def write_segment(root, name, meta, frames=(), clouds=(), odom=()) -> Path:
    """frames: (stamp_ns, image); clouds: (stamp_ns, Nx3, frame_id); odom: (stamp_ns, T_odom_base, wz)."""
    seg = Path(root) / name
    (seg / "frames").mkdir(parents=True, exist_ok=True)
    (seg / "clouds").mkdir(exist_ok=True)
    cio.write_yaml(seg / "meta.yaml", {"segment": name, **meta})
    with open(seg / "frames.csv", "w") as f:
        f.write("stamp_ns,file,width,height\n")
        for stamp, img in frames:
            rel = f"frames/{stamp}.jpg"
            cv2.imwrite(str(seg / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            f.write(f"{stamp},{rel},{img.shape[1]},{img.shape[0]}\n")
    with open(seg / "clouds.csv", "w") as f:
        f.write("stamp_ns,file,frame_id,n_points\n")
        for stamp, pts, frame in clouds:
            rel = f"clouds/{stamp}.npy"
            np.save(seg / rel, np.asarray(pts, np.float32))
            f.write(f"{stamp},{rel},{frame},{len(pts)}\n")
    with open(seg / "odom.csv", "w") as f:
        f.write("stamp_ns,x,y,z,qx,qy,qz,qw,wz\n")
        for stamp, t, wz in odom:
            q = quaternion_from_matrix(t[:3, :3])
            f.write(f"{stamp},{t[0, 3]},{t[1, 3]},{t[2, 3]},{q[0]},{q[1]},{q[2]},{q[3]},{wz}\n")
    return seg


# ------------------------------------------------------------------ ray casting (base frame boxes)
def raycast(origin: np.ndarray, dirs: np.ndarray, boxes) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hit distance and box index per ray against axis-aligned boxes (inf / -1 = miss)."""
    best_t = np.full(dirs.shape[0], np.inf)
    best_i = np.full(dirs.shape[0], -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        for i, (lo, hi, _) in enumerate(boxes):
            t1 = (np.asarray(lo) - origin) * inv
            t2 = (np.asarray(hi) - origin) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0)
            closer = hit & (tmin < best_t)
            best_t[closer] = tmin[closer]
            best_i[closer] = i
    return best_t, best_i


SCENE_BOXES = [
    ((4.5, -6.0, -1.0), (4.7, 6.0, 2.5), 190),  # back wall
    ((1.8, -0.7, -0.35), (2.3, -0.1, 0.35), 60),  # box A
    ((2.6, 0.4, -0.3), (3.0, 0.9, 0.6), 120),  # box B
    ((1.4, 0.9, -0.4), (1.5, 1.0, 1.2), 30),  # pole
    ((3.2, -1.6, 0.25), (3.5, -1.1, 0.7), 150),  # shelf box
]


def render_view(cam: cio.CameraModel, base_from_cam: np.ndarray, boxes=SCENE_BOXES, background=235) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
    pix = np.column_stack([us.ravel(), vs.ravel()]).astype(np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pix, cam.k, cam.d).reshape(-1, 2)
    dirs_cam = np.column_stack([norm, np.ones(norm.shape[0])])
    dirs = dirs_cam @ base_from_cam[:3, :3].T
    _, idx = raycast(base_from_cam[:3, 3], dirs, boxes)
    shade = np.array([b[2] for b in boxes] + [background], np.uint8)
    img = shade[idx].reshape(cam.height, cam.width)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def lidar_scan(origin=(0.29, 0.0, -0.05), boxes=SCENE_BOXES, elev=(-15.0, 25.0, 0.6), azim=(-70.0, 70.0, 0.25),
               noise_m=0.003, seed=0) -> np.ndarray:
    el = np.radians(np.arange(*elev))
    az = np.radians(np.arange(*azim))
    e, a = np.meshgrid(el, az, indexing="ij")
    dirs = np.column_stack([(np.cos(e) * np.cos(a)).ravel(), (np.cos(e) * np.sin(a)).ravel(), np.sin(e).ravel()])
    o = np.asarray(origin, np.float64)
    t, idx = raycast(o, dirs, boxes)
    hit = idx >= 0
    t = t[hit] + np.random.default_rng(seed).normal(0.0, noise_m, hit.sum())
    return o + dirs[hit] * t[:, None]


def pose(x=0.0, y=0.0, yaw=0.0) -> np.ndarray:
    return cio.make_transform(cio.rpy_to_matrix(0.0, 0.0, yaw), [x, y, 0.0])


def scene_capture(root, true_xyz, true_rpy, meta_xyz, meta_rpy, cam_dict, robot=(1.0, 2.0, 0.5), taped=None,
                  n_frames=3, n_clouds=6) -> Path:
    """Static robot facing SCENE_BOXES; clouds stored in odom; image rendered with the TRUE extrinsic."""
    cam = cio.camera_model_from_dict(cam_dict)
    odom_from_base = pose(*robot)
    pts_odom = cio.transform_points(odom_from_base, lidar_scan())
    img = render_view(cam, cio.base_from_camera(true_xyz, true_rpy))
    t0 = 1_000_000_000_000
    frames = [(t0 + 200_000_000 + i * 100_000_000, img) for i in range(n_frames)]
    clouds = [(t0 + i * 100_000_000, pts_odom, "odom") for i in range(n_clouds)]
    odom = [(t0 + i * 20_000_000, odom_from_base, 0.0) for i in range(60)]
    meta = {"camera_frame": "front_camera_optical_frame", "latency_s": 0.0, "intrinsics": cam_dict,
            "extrinsics": extrinsics_dict(meta_xyz, meta_rpy)}
    if taped is not None:
        meta["taped_objects"] = taped
    return write_segment(root, "scene", meta, frames, clouds, odom)
