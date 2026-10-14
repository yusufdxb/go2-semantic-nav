"""Real SQLite bag checks for complete, atomic fusion measurement exports."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml
from rosbags.rosbag2 import Writer

spec = importlib.util.spec_from_file_location(
    "export_measurements", Path(__file__).with_name("export_measurements.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bag = self.root / "bag"
        self.output = self.root / "measurements.jsonl"
        self.store = m.fusion_typestore()

    def obj(self, valid=True, x=1.0):
        point = self.store.types["geometry_msgs/msg/Point"](x, 2.0, 3.0)
        return self.store.types["go2_fusion_cpp/msg/FusionObject"](
            "chair", 0.9, 0 if valid else 6, "", valid, 7 if valid else 0,
            12 if valid else 0, point, np.zeros(9, dtype=np.float64),
        )

    def array(self, objects, status=0):
        stamp = self.store.types["builtin_interfaces/msg/Time"](10, 123)
        header = self.store.types["std_msgs/msg/Header"](stamp, "base_link")
        return self.store.types[m.MESSAGE_TYPE](
            header, stamp, "camera", "lidar", "measured-v1", status, "", objects,
            0.1, 0.2, 0.0, -1.0, False, 0.0, 0,
        )

    def write_bag(self, arrays):
        with Writer(self.bag, version=9) as writer:
            connection = writer.add_connection("/fusion/objects", m.MESSAGE_TYPE, typestore=self.store)
            other = writer.add_connection("/other", "std_msgs/msg/String", typestore=self.store)
            for index, array in enumerate(arrays):
                writer.write(connection, index + 1, self.store.serialize_cdr(array, m.MESSAGE_TYPE))
            value = self.store.types["std_msgs/msg/String"]("complete-stream-check")
            writer.write(other, len(arrays) + 1, self.store.serialize_cdr(value, "std_msgs/msg/String"))

    def test_real_bag_preserves_indices_invalid_and_empty_arrays(self):
        self.write_bag([
            self.array([self.obj(False, float("nan")), self.obj()]),
            self.array([], status=1),
        ])
        summary = m.export_measurements(self.bag, self.output)
        records = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(summary, {"arrays": 2, "objects": 2, "valid_objects": 1, "bag_messages": 3})
        self.assertEqual(records[0]["stamp_ns"], 10_000_000_123)
        self.assertEqual(records[0]["frame_id"], "base_link")
        self.assertEqual(records[0]["calibration_id"], "measured-v1")
        self.assertEqual(records[0]["objects"][0]["position_m"], None)
        self.assertEqual(records[0]["objects"][1]["index"], 1)
        self.assertEqual(records[0]["objects"][1]["position_m"], [1.0, 2.0, 3.0])
        self.assertEqual(records[1]["array_status"], 1)
        self.assertEqual(records[1]["objects"], [])

    def test_nonfinite_valid_position_keeps_existing_output(self):
        self.write_bag([self.array([self.obj()]), self.array([self.obj(x=float("inf"))])])
        self.output.write_text("previous output\n")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.export_measurements(self.bag, self.output)
        self.assertEqual(self.output.read_text(), "previous output\n")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_metadata_count_mismatch_is_rejected(self):
        self.write_bag([self.array([self.obj()])])
        metadata_path = self.bag / "metadata.yaml"
        metadata = yaml.safe_load(metadata_path.read_text())
        metadata["rosbag2_bagfile_information"]["message_count"] += 1
        metadata_path.write_text(yaml.safe_dump(metadata))
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            m.export_measurements(self.bag, self.output)
        self.assertFalse(self.output.exists())

    def test_truncated_database_is_rejected_without_salvage(self):
        self.write_bag([self.array([self.obj()])])
        database = next(self.bag.glob("*.db3"))
        with database.open("r+b") as stream:
            stream.truncate(database.stat().st_size - 4096)
        with self.assertRaises((ValueError, m.sqlite3.Error)):
            m.export_measurements(self.bag, self.output)
        self.assertFalse(self.output.exists())

    def test_output_inside_recording_is_rejected(self):
        self.write_bag([self.array([])])
        with self.assertRaisesRegex(ValueError, "outside"):
            m.export_measurements(self.bag, self.bag / "measurements.jsonl")


if __name__ == "__main__":
    unittest.main()
