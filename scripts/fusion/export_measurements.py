#!/usr/bin/env python3
"""Export fusion measurements from an intact ROS 2 SQLite bag without ROS.

Requires rosbags and PyYAML. Every listed database and the complete message
stream must pass integrity/count checks before the JSONL output is published.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path

import yaml
from rosbags.highlevel import AnyReader, AnyReaderError
from rosbags.typesys import Stores, TypesysError, get_types_from_msg, get_typestore

MESSAGE_TYPE = "go2_fusion_cpp/msg/FusionArray"


def fusion_typestore():
    store = get_typestore(Stores.ROS2_HUMBLE)
    definitions = {}
    message_dir = Path(__file__).resolve().parents[2] / "ros2_ws/src/go2_fusion_cpp/msg"
    for name in ("FusionObject", "FusionArray"):
        definitions.update(
            get_types_from_msg((message_dir / f"{name}.msg").read_text(), f"go2_fusion_cpp/msg/{name}")
        )
    store.register(definitions)
    return store


def verify_bag(bag: Path):
    if not bag.is_dir():
        raise ValueError("--bag must be a complete ROS 2 bag directory with metadata.yaml")
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    if metadata["storage_identifier"] != "sqlite3":
        raise ValueError("only ROS 2 sqlite3 bags are supported")
    if metadata.get("compression_mode", "") not in ("", "none"):
        raise ValueError("compressed bags must be converted to an intact uncompressed bag first")
    paths = metadata["relative_file_paths"]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("bag metadata must list distinct SQLite files")
    total = 0
    for relative in paths:
        database = (bag / relative).resolve()
        if not database.is_relative_to(bag) or database.suffix != ".db3" or not database.is_file():
            raise ValueError("missing or invalid SQLite file in bag metadata")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchall()
            if result != [("ok",)]:
                raise ValueError("SQLite integrity check failed")
            total += connection.execute("SELECT count(*) FROM messages").fetchone()[0]
    declared = int(metadata["message_count"])
    if total != declared:
        raise ValueError(f"bag message count mismatch: metadata={declared}, SQLite={total}")
    return declared


def measurement_record(message):
    objects = []
    for index, obj in enumerate(message.objects):
        valid = bool(obj.valid)
        position = [float(obj.position.x), float(obj.position.y), float(obj.position.z)]
        if valid and not all(math.isfinite(value) for value in position):
            raise ValueError(f"valid object {index} has a nonfinite position")
        objects.append({
            "index": index,
            "label": str(obj.label),
            "status": int(obj.status),
            "valid": valid,
            "track_id": int(obj.track_id),
            "support_points": int(obj.support_points),
            "position_m": position if valid else None,
        })
    return {
        "stamp_ns": int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec),
        "frame_id": str(message.header.frame_id),
        "calibration_id": str(message.calibration_id),
        "array_status": int(message.status),
        "objects": objects,
    }


def export_measurements(bag: Path, output: Path, topic: str = "/fusion/objects"):
    bag, output = bag.resolve(), output.resolve()
    if output.is_relative_to(bag):
        raise ValueError("output must be outside the source recording directory")
    expected = verify_bag(bag)
    store = fusion_typestore()
    arrays = objects = valid_objects = consumed = 0
    temporary = None
    try:
        with AnyReader([bag], default_typestore=store) as reader:
            selected = [connection for connection in reader.connections if connection.topic == topic]
            if not selected:
                raise ValueError(f"bag has no {topic} topic")
            if any(connection.msgtype != MESSAGE_TYPE for connection in selected):
                raise ValueError(f"{topic} must contain {MESSAGE_TYPE}")
            # Embedded definitions are validated against the repository schema.
            reader.typestore.register(store.fielddefs)
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent,
                prefix=f".{output.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for connection, _, raw in reader.messages():
                    consumed += 1
                    if connection.topic != topic:
                        continue
                    record = measurement_record(reader.deserialize(raw, connection.msgtype))
                    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                    arrays += 1
                    objects += len(record["objects"])
                    valid_objects += sum(obj["valid"] for obj in record["objects"])
                if consumed != expected or consumed != reader.message_count:
                    raise ValueError("complete bag stream does not match declared message count")
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"arrays": arrays, "objects": objects, "valid_objects": valid_objects, "bag_messages": consumed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/fusion/objects")
    args = parser.parse_args()
    try:
        summary = export_measurements(args.bag, args.output, args.topic)
    except (OSError, ValueError, KeyError, sqlite3.Error, yaml.YAMLError, AnyReaderError, TypesysError) as error:
        parser.exit(1, f"export failed: {error}\n")
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
