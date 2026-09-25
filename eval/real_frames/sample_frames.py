#!/usr/bin/env python3
"""Sample real GO2 front-camera frames on a fixed time grid (no content filtering).

Reads rosbag2 sqlite files directly, so it needs rclpy only for deserialisation.
Writes <out>/<bag_tag>_t<sec>.png plus manifest.csv (bag, frame stamp, grid time).
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

TOPIC = "/camera/front/image_raw"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", action="append", required=True, help="tag=path/to/bag_0.db3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-s", type=float, default=15.0)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.bag:
        tag, path = spec.split("=", 1)
        db = sqlite3.connect(path)
        (tid,) = db.execute("select id from topics where name=?", (TOPIC,)).fetchone()
        msgs = db.execute(
            "select timestamp, data from messages where topic_id=? order by timestamp", (tid,)
        ).fetchall()
        stamps = np.array([m[0] for m in msgs], dtype=np.int64)
        t0 = stamps[0]
        grid_s = 0.0
        while t0 + int(grid_s * 1e9) <= stamps[-1]:
            i = int(np.argmin(np.abs(stamps - (t0 + int(grid_s * 1e9)))))
            msg = deserialize_message(msgs[i][1], Image)
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == "rgb8":
                img = img[..., ::-1]
            name = f"{tag}_t{int(grid_s):04d}.png"
            cv2.imwrite(str(out / name), img)
            rows.append({"frame": name, "bag": tag, "block": f"{tag}_b{int(grid_s // 60)}",
                         "grid_s": grid_s, "bag_stamp_ns": int(stamps[i])})
            grid_s += args.step_s
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
