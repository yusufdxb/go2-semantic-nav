#!/usr/bin/env python3
"""Audit a ROS 2 SQLite recording and extract bounded binary fusion replay inputs.

Requires numpy, PyYAML and rosbags; OpenCV is optional for diagnostic overlays.
Never repairs or writes the source recording. Nominal geometry stays untrusted.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import struct
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import yaml
from rosbags.typesys import Stores, get_typestore

MAGIC = b"G2FUS001"
MAX_POINTS = 250_000


def stamp(h):
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def quat_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(q @ q)
    if not np.isfinite(q).all() or n < 1e-12:
        raise ValueError("invalid TF quaternion")
    x, y, z, w = q / np.sqrt(n)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def tf_values(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])


def matrix(t, q):
    out = np.eye(4)
    out[:3, :3] = quat_matrix(q)
    out[:3, 3] = t
    if not np.isfinite(out).all():
        raise ValueError("nonfinite TF")
    return out


def interpolate_tf(samples, times, ns, max_gap_ns):
    j = int(np.searchsorted(times, ns))
    if j < len(times) and times[j] == ns:
        return matrix(*samples[j][1:])
    if j == 0 or j == len(times):
        raise ValueError("missing bracketing TF")
    a, b = samples[j - 1], samples[j]
    if b[0] - a[0] > max_gap_ns:
        raise ValueError("TF interpolation gap")
    f = (ns - a[0]) / (b[0] - a[0])
    qa = a[2] / np.linalg.norm(a[2])
    qb = b[2] / np.linalg.norm(b[2])
    dot = float(qa @ qb)
    if dot < 0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        q = qa + f * (qb - qa)
    else:
        theta = math.acos(min(1.0, dot))
        q = (math.sin((1 - f) * theta) * qa + math.sin(f * theta) * qb) / math.sin(theta)
    return matrix(a[1] + f * (b[1] - a[1]), q)


def cloud_xyz(msg):
    fields = {f.name: f for f in msg.fields}
    if not {"x", "y", "z"} <= fields.keys():
        raise ValueError("missing XYZ")
    if (
        msg.point_step <= 0
        or msg.row_step < msg.width * msg.point_step
        or len(msg.data) != msg.row_step * msg.height
    ):
        raise ValueError("malformed cloud layout")
    if msg.width * msg.height > MAX_POINTS:
        raise ValueError("cloud point capacity exceeded")
    endian = ">" if msg.is_bigendian else "<"
    arrays = []
    for name in ("x", "y", "z"):
        f = fields[name]
        if f.datatype not in (7, 8) or f.count != 1:
            raise ValueError("unsupported XYZ field")
        dtype = np.dtype(endian + ("f4" if f.datatype == 7 else "f8"))
        if f.offset < 0 or f.offset + dtype.itemsize > msg.point_step:
            raise ValueError("XYZ offset outside point")
        arrays.append(
            np.ndarray(
                (msg.height, msg.width),
                dtype=dtype,
                buffer=msg.data,
                offset=f.offset,
                strides=(msg.row_step, msg.point_step),
            ).ravel()
        )
    xyz = np.column_stack(arrays).astype("<f4")
    # Keep invalid points in the replay. Runtime validation must reject them.
    return xyz


def inspect_source(source, output, recover, recovery_copy=None):
    with source.open("rb") as f:
        header = f.read(100)
    if header[:16] != b"SQLite format 3\x00":
        raise ValueError("not SQLite 3")
    page_size = struct.unpack(">H", header[16:18])[0] or 65536
    if page_size == 1:
        page_size = 65536
    if page_size < 512 or page_size > 65536 or page_size & (page_size - 1):
        raise ValueError("invalid SQLite page size")
    size = source.stat().st_size
    declared = struct.unpack(">I", header[28:32])[0]
    result = {
        "source_bytes": size,
        "page_size": page_size,
        "declared_pages": declared,
        "available_pages": size // page_size,
        "source_complete": size == declared * page_size,
        "partial_recovery": False,
    }
    try:
        c = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
        result["quick_check"] = c.execute("pragma quick_check").fetchall()
        c.close()
    except sqlite3.DatabaseError as exc:
        result["quick_check_error"] = str(exc)
    if result.get("quick_check") == [("ok",)]:
        return source, result
    if not recover:
        raise ValueError(
            "recording fails integrity check; use --recover-prefix to write a separate diagnostic copy"
        )
    if size % page_size:
        raise ValueError("incomplete final SQLite page; explicit page salvage is unsupported")
    derived = recovery_copy or output / "readable_prefix.db3"
    if derived.resolve() == source.resolve() or derived.resolve().parent == source.resolve().parent:
        raise ValueError("derived copy must be outside the recording directory")
    derived.parent.mkdir(parents=True, exist_ok=True)
    # Only a derived copy gets a bounded page count. This is not full recovery.
    shutil.copyfile(source, derived)
    with derived.open("r+b") as f:
        f.seek(28)
        f.write(struct.pack(">I", size // page_size))
    result["partial_recovery"] = True
    return derived, result


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {}
    return dict(
        zip(("min", "p50", "p95", "p99", "max"), map(float, np.percentile(a, [0, 50, 95, 99, 100])))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--recover-prefix", action="store_true")
    parser.add_argument(
        "--recovery-copy", type=Path, help="explicit derived copy outside the recording directory"
    )
    parser.add_argument("--cloud-topic", default="/go2/lidar/points")
    parser.add_argument("--image-topic", default="/camera/front/image_raw")
    parser.add_argument("--accumulate-s", type=float, default=0.0)
    parser.add_argument("--max-cloud-skew-ms", type=float, default=100.0)
    parser.add_argument("--max-tf-gap-ms", type=float, default=100.0)
    parser.add_argument("--visualizations", type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.visualizations <= 32:
        raise ValueError("visualization count outside [0,32]")
    if not all(math.isfinite(x) and x > 0 for x in (args.max_cloud_skew_ms, args.max_tf_gap_ms)):
        raise ValueError("timestamp tolerances must be finite and positive")
    if args.output.resolve() == args.bag.parent.resolve():
        raise ValueError("output must be separate from recording")
    if not 0 <= args.accumulate_s <= 1.0:
        raise ValueError("accumulation must be within [0,1] second")
    args.output.mkdir(parents=True, exist_ok=True)
    db, audit = inspect_source(args.bag, args.output, args.recover_prefix, args.recovery_copy)
    c = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    topics = {int(r[0]): (r[1], r[2]) for r in c.execute("select id,name,type from topics")}
    store = get_typestore(Stores.ROS2_HUMBLE)
    stats = defaultdict(
        lambda: {
            "count": 0,
            "record_ns": [],
            "header_ns": [],
            "frames": set(),
            "sizes": set(),
            "point_counts": [],
        }
    )
    images = []
    clouds = []
    edges = defaultdict(list)
    static = {}
    camera_infos = []
    camera_models = set()
    deserialize_errors = 0
    last_id = 0
    try:
        for mid, tid, record_ns, data in c.execute(
            "select id,topic_id,timestamp,data from messages not indexed order by id"
        ):
            last_id = mid
            topic, typ = topics[tid]
            s = stats[topic]
            s["count"] += 1
            s["record_ns"].append(record_ns)
            try:
                msg = store.deserialize_cdr(data, typ)
            except Exception:
                deserialize_errors += 1
                continue
            if hasattr(msg, "header"):
                ns = stamp(msg.header)
                s["header_ns"].append(ns)
                s["frames"].add(msg.header.frame_id)
                if hasattr(msg, "width"):
                    s["sizes"].add((int(msg.width), int(msg.height)))
            if typ == "tf2_msgs/msg/TFMessage":
                for tf in msg.transforms:
                    key = (tf.header.frame_id, tf.child_frame_id)
                    vals = tf_values(tf)
                    if topic == "/tf_static":
                        static[key] = matrix(*vals)
                    else:
                        edges[key].append((stamp(tf.header), *vals))
            elif typ == "sensor_msgs/msg/CameraInfo":
                camera_models.add(
                    (
                        int(msg.width),
                        int(msg.height),
                        msg.distortion_model,
                        tuple(msg.k),
                        tuple(msg.d),
                    )
                )
                if len(camera_infos) < 8:
                    camera_infos.append(
                        {
                            "stamp_ns": ns,
                            "width": int(msg.width),
                            "height": int(msg.height),
                            "frame": msg.header.frame_id,
                            "K": msg.k.tolist(),
                            "D": msg.d.tolist(),
                            "model": msg.distortion_model,
                        }
                    )
            if topic == args.image_topic:
                images.append((ns, mid, msg.header.frame_id, int(msg.width), int(msg.height)))
            if topic == args.cloud_topic:
                s["point_counts"].append(int(msg.width) * int(msg.height))
                clouds.append((ns, mid, msg.header.frame_id))
    except sqlite3.DatabaseError as exc:
        audit["prefix_stop_error"] = str(exc)
    audit["last_readable_message_id"] = last_id
    audit["deserialize_errors"] = deserialize_errors
    audit["topics"] = {}
    for topic, s in stats.items():
        rn = np.asarray(s["record_ns"], dtype=np.int64)
        hn = np.asarray(s["header_ns"], dtype=np.int64)
        audit["topics"][topic] = {
            "count": s["count"],
            "first_record_ns": int(rn.min()),
            "last_record_ns": int(rn.max()),
            "record_interval_ms": summary(np.diff(rn) / 1e6),
            "frames": sorted(s["frames"]),
            "sizes": sorted(s["sizes"]),
            "point_counts": summary(s["point_counts"]),
        }
        if hn.size == rn.size:
            audit["topics"][topic]["record_minus_header_ms"] = summary((rn - hn) / 1e6)
            audit["topics"][topic]["header_interval_ms"] = summary(np.diff(hn) / 1e6)
    audit["camera_info_samples"] = camera_infos
    audit["unique_camera_info_models"] = len(camera_models)
    audit["tf_static"] = {f"{p}->{ch}": m.tolist() for (p, ch), m in static.items()}
    audit["tf_dynamic"] = {
        f"{p}->{ch}": {
            "count": len(v),
            "first_ns": min(t[0] for t in v),
            "last_ns": max(t[0] for t in v),
        }
        for (p, ch), v in edges.items()
    }
    intrinsic = yaml.safe_load(args.intrinsics.read_text())
    K = intrinsic["camera_matrix"]["data"]
    D = intrinsic["distortion_coefficients"]["data"]
    cam = {
        "width": intrinsic["image_width"],
        "height": intrinsic["image_height"],
        "fx": K[0],
        "fy": K[4],
        "cx": K[2],
        "cy": K[5],
        "distortion": D,
        "frame": "front_camera_optical_frame",
        "calibration_valid": False,
        "calibration_provenance": "diagnostic only; independently measured intrinsics and extrinsics not supplied",
    }
    manifest = {
        "format": "G2FUS001",
        "camera": cam,
        "geometry_trusted": False,
        "transform_convention": "camera_from_cloud, R row-major 9 doubles then translation 3 doubles",
        "association": "one full-image diagnostic ROI per frame; no detector or ground-truth object claim",
        "accumulate_s": args.accumulate_s,
        "timestamp_note": "header timestamps retained; record-header differences are receipt offsets, not sensor capture latency",
        "partial_recording": audit["partial_recovery"],
    }
    (args.output / "camera.json").write_text(json.dumps(cam, indent=2) + "\n")
    tf_samples = sorted(edges.get(("odom", "base_link"), []), key=lambda x: x[0])
    tf_times = np.array([t[0] for t in tf_samples])
    base_camera = static.get(("base_link", cam["frame"]))
    images.sort()
    clouds.sort()
    cloud_times = np.array([x[0] for x in clouds], dtype=np.int64)
    cache = OrderedDict()
    written = 0
    rejected = defaultdict(int)
    skews = []
    point_counts = []
    visual_ids = set(np.linspace(0, max(0, len(images) - 1), args.visualizations, dtype=int))

    def read_cloud(item):
        mid = item[1]
        if mid in cache:
            cache.move_to_end(mid)
            return cache[mid]
        raw = c.execute("select data from messages where id=?", (mid,)).fetchone()[0]
        out = cloud_xyz(
            store.deserialize_cdr(
                raw, topics[next(k for k, v in topics.items() if v[0] == args.cloud_topic)][1]
            )
        )
        cache[mid] = out
        while len(cache) > 32:
            cache.popitem(last=False)
        return out

    temporary_replay = args.output / "replay.g2f.tmp"
    with temporary_replay.open("wb") as f:
        f.write(MAGIC + struct.pack("<I", 0))
        for i, (image_ns, mid, frame, width, height) in enumerate(images):
            try:
                if (frame, width, height) != (cam["frame"], cam["width"], cam["height"]):
                    raise ValueError("image geometry mismatch")
                if base_camera is None:
                    raise ValueError("missing camera extrinsic")
                if not clouds:
                    raise ValueError("no clouds")
                j = int(np.searchsorted(cloud_times, image_ns))
                j = min(
                    range(max(0, j - 1), min(len(clouds), j + 1)),
                    key=lambda k: abs(clouds[k][0] - image_ns),
                )
                cloud_ns = clouds[j][0]
                if abs(cloud_ns - image_ns) > args.max_cloud_skew_ms * 1e6:
                    raise ValueError("cloud timestamp skew")
                if clouds[j][2] != "odom":
                    raise ValueError("unsupported cloud frame")
                odom_base = interpolate_tf(tf_samples, tf_times, image_ns, args.max_tf_gap_ms * 1e6)
                camera_cloud = np.linalg.inv(base_camera) @ np.linalg.inv(odom_base)
                if args.accumulate_s:
                    start = int(
                        np.searchsorted(cloud_times, image_ns - int(args.accumulate_s * 1e9))
                    )
                    end = int(np.searchsorted(cloud_times, image_ns, side="right"))
                    parts = [read_cloud(x) for x in clouds[start:end]]
                    if not parts:
                        raise ValueError("empty accumulation window")
                    cloud_ns = clouds[end - 1][0]
                    xyz = np.concatenate(parts)
                else:
                    xyz = read_cloud(clouds[j])
                if len(xyz) > MAX_POINTS:
                    raise ValueError("accumulated point capacity exceeded")
                f.write(struct.pack("<qqI", image_ns, cloud_ns, len(xyz)))
                f.write(struct.pack("<12d", *camera_cloud[:3, :3].ravel(), *camera_cloud[:3, 3]))
                f.write(np.asarray(xyz, dtype="<f4").tobytes())
                f.write(struct.pack("<I5fI", 1, 0.0, 0.0, float(width), float(height), 0.0, 0))
                written += 1
                skews.append((cloud_ns - image_ns) / 1e6)
                point_counts.append(len(xyz))
                if i in visual_ids:
                    raw = c.execute("select data from messages where id=?", (mid,)).fetchone()[0]
                    msg = store.deserialize_cdr(raw, "sensor_msgs/msg/Image")
                    save_visual(args.output, written, msg, xyz, camera_cloud, cam)
            except (ValueError, sqlite3.DatabaseError) as exc:
                rejected[str(exc)] += 1
        f.seek(8)
        f.write(struct.pack("<I", written))
    temporary_replay.replace(args.output / "replay.g2f")
    manifest.update(
        {
            "frame_count": written,
            "rejected_frames": dict(rejected),
            "cloud_minus_image_ms": summary(skews),
            "points_per_frame": summary(point_counts),
        }
    )
    audit["calibration_trust"] = (
        "invalid: recorded geometry matches explicit nominal configuration; raw LiDAR clock differs from normalized clock"
    )
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    c.close()


def save_visual(output, index, msg, xyz, tf, cam):
    try:
        import cv2
    except ImportError:
        return
    if msg.encoding not in ("bgr8", "rgb8") or msg.step < msg.width * 3:
        return
    rgb = np.ndarray(
        (msg.height, msg.width, 3), np.uint8, buffer=msg.data, strides=(msg.step, 3, 1)
    ).copy()
    if msg.encoding == "rgb8":
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    p = xyz.astype(np.float64) @ tf[:3, :3].T + tf[:3, 3]
    p = p[np.isfinite(p).all(axis=1) & (p[:, 2] > 0.2) & (p[:, 2] < 8.0)]
    x = p[:, 0] / p[:, 2]
    y = p[:, 1] / p[:, 2]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = cam["distortion"]
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    u = cam["fx"] * (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cam["cx"]
    v = cam["fy"] * (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cam["cy"]
    valid = (r2 <= 1.3**2) & (u >= 0) & (u < msg.width) & (v >= 0) & (v < msg.height)
    u, v, z = u[valid], v[valid], p[valid, 2]
    for uu, vv, zz in zip(u, v, z):
        t = float(np.clip((zz - 0.2) / 7.8, 0, 1))
        cv2.circle(
            rgb,
            (int(round(uu)), int(round(vv))),
            1,
            (int(255 * t), int(255 * (1 - abs(2 * t - 1))), int(255 * (1 - t))),
            -1,
        )
    cv2.rectangle(rgb, (0, 0), (msg.width, 66), (0, 0, 0), -1)
    cv2.putText(
        rgb,
        "NOMINAL GEOMETRY: DIAGNOSTIC ONLY, NO VALID POSITIONS",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        rgb,
        f"{len(xyz)} cloud points, {len(z)} projected; RGB stamp {stamp(msg.header)}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output / f"projection_{index:04d}.jpg"), rgb)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, sqlite3.DatabaseError) as exc:
        sys.exit(str(exc))
