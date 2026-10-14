#!/usr/bin/env python3
"""Sample a replay with the existing detector backend, preserving fusion geometry.

Produces measured detector boxes, not ground truth. Geometry stays untrusted.
Detector timings are offline inference wall time, separate from fusion compute.
"""

import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from check_projection import frames
from extract_replay import stamp, summary
from rosbags.typesys import Stores, get_typestore

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ros2_ws/src/go2_open_vocab_detector"))


def main():
    from go2_open_vocab_detector.backends.ultralytics_detector import UltralyticsOpenVocabDetector

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--readable-bag", type=Path, required=True)
    p.add_argument("--detector-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--last-readable-message-id", type=int, required=True)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if not 1 <= a.samples <= 1000:
        raise ValueError("samples outside [1,1000]")
    a.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(a.detector_config.read_text())["go2_open_vocab_detector"][
        "ros__parameters"
    ]
    model = UltralyticsOpenVocabDetector(config["detector_backend"])
    model.load(a.device, config["prompts"])
    c = sqlite3.connect(f"file:{a.readable_bag.resolve()}?mode=ro", uri=True)
    store = get_typestore(Stores.ROS2_HUMBLE)
    tid = c.execute("select id from topics where name='/camera/front/image_raw'").fetchone()[0]
    image_ids = {}
    for mid, raw in c.execute(
        "select id,data from messages where id<=? and topic_id=? order by id",
        (a.last_readable_message_id, tid),
    ):
        msg = store.deserialize_cdr(raw, "sensor_msgs/msg/Image")
        image_ids[stamp(msg.header)] = mid
    with a.replay.open("rb") as f:
        if f.read(8) != b"G2FUS001":
            raise ValueError("invalid replay")
        count = struct.unpack("<I", f.read(4))[0]
    selected = set(np.linspace(0, count - 1, min(count, a.samples), dtype=int))
    latencies = []
    outputs = []
    label_ids = {}
    warm = False
    temporary = a.output / "replay.g2f.tmp"
    with temporary.open("wb") as f:
        f.write(b"G2FUS001" + struct.pack("<I", len(selected)))
        for idx, image_ns, cloud_ns, xyz, transform, _ in frames(a.replay):
            if idx not in selected:
                continue
            raw = c.execute(
                "select data from messages where id=?", (image_ids[image_ns],)
            ).fetchone()[0]
            msg = store.deserialize_cdr(raw, "sensor_msgs/msg/Image")
            if msg.encoding not in ("bgr8", "rgb8"):
                raise ValueError("unsupported image encoding")
            bgr = np.ndarray(
                (msg.height, msg.width, 3), np.uint8, buffer=msg.data, strides=(msg.step, 3, 1)
            ).copy()
            if msg.encoding == "rgb8":
                bgr = bgr[:, :, ::-1].copy()
            if not warm:
                model.detect(
                    bgr, float(config["conf_threshold"]), int(config["max_objects_per_frame"])
                )
                warm = True
            t = time.perf_counter_ns()
            det = model.detect(
                bgr, float(config["conf_threshold"]), int(config["max_objects_per_frame"])
            )
            wall_ms = (time.perf_counter_ns() - t) / 1e6
            latencies.append(wall_ms)
            f.write(struct.pack("<qqI", image_ns, cloud_ns, len(xyz)))
            f.write(struct.pack("<12d", *transform))
            f.write(np.asarray(xyz, dtype="<f4").tobytes())
            f.write(struct.pack("<I", len(det.labels)))
            boxes = []
            for box, score, label in zip(det.boxes_xyxy, det.scores, det.labels):
                lid = label_ids.setdefault(label, len(label_ids) + 1)
                box = box.copy()
                box[[0, 2]] = np.clip(box[[0, 2]], 0, msg.width)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, msg.height)
                f.write(struct.pack("<5fI", *map(float, box), float(score), lid))
                boxes.append(
                    {"xyxy": box.tolist(), "score": float(score), "label": label, "label_id": lid}
                )
            outputs.append(
                {
                    "source_frame": idx,
                    "image_ns": image_ns,
                    "detector_wall_ms": wall_ms,
                    "boxes": boxes,
                }
            )
            if len(outputs) <= 6:
                import cv2

                for d in boxes:
                    x0, y0, x1, y1 = map(int, d["xyxy"])
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(
                        bgr,
                        f"{d['label']} {d['score']:.2f}",
                        (x0, max(18, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(bgr, (0, 0), (msg.width, 35), (0, 0, 0), -1)
                cv2.putText(
                    bgr,
                    "RECORDED DETECTOR BOXES, NOMINAL GEOMETRY, NO VALID POSITIONS",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(a.output / f"detections_{idx:04d}.jpg"), bgr)
    temporary.replace(a.output / "replay.g2f")
    report = {
        "geometry_trusted": False,
        "metric_accuracy_validated": False,
        "detector_backend": config["detector_backend"],
        "device": a.device,
        "samples": len(outputs),
        "detections": sum(len(x["boxes"]) for x in outputs),
        "label_ids": label_ids,
        "detector_wall_ms": summary(latencies),
        "sensor_to_output_ms": None,
        "timing_scope": "offline detector wall time after one warmup, including host result retrieval; excludes segmentation, embedding, sensor capture, fusion and publication",
        "association": "actual detector bounding boxes; no segmentation masks or ground-truth object positions",
        "frames": outputs,
    }
    (a.output / "detector.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    c.close()


if __name__ == "__main__":
    main()
