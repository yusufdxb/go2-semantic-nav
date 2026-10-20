"""Independent layout and rigid-transform checks for recorded replay extraction."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

spec = importlib.util.spec_from_file_location(
    "extract_replay", Path(__file__).with_name("extract_replay.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExtractionTests(unittest.TestCase):
    def test_padded_organized_big_endian_cloud(self):
        data = bytearray(72)
        expected = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
        for row in range(2):
            for col in range(2):
                a = np.ndarray((3,), dtype=">f4", buffer=data, offset=row * 36 + col * 16)
                a[:] = expected[row * 2 + col]
        fields = [
            NS(name=n, offset=o, datatype=7, count=1) for n, o in zip(("x", "y", "z"), (0, 4, 8))
        ]
        msg = NS(
            width=2,
            height=2,
            point_step=16,
            row_step=36,
            data=data,
            is_bigendian=True,
            fields=fields,
        )
        np.testing.assert_array_equal(m.cloud_xyz(msg), expected)
        msg.row_step = 31
        with self.assertRaisesRegex(ValueError, "malformed"):
            m.cloud_xyz(msg)

    def test_interpolation_preserves_rigid_transform_and_antipodal_quaternion(self):
        q = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
        samples = [(100, np.zeros(3), q), (200, np.array([2.0, 4.0, 6.0]), -q)]
        actual = m.interpolate_tf(samples, np.array([100, 200]), 150, 100)
        np.testing.assert_allclose(actual[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(
            actual[:3, :3] @ np.array([1.0, 0.0, 0.0]), [0, 1, 0], atol=1e-14
        )
        np.testing.assert_allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-14)
        with self.assertRaisesRegex(ValueError, "gap"):
            m.interpolate_tf(samples, np.array([100, 200]), 150, 99)
        with self.assertRaisesRegex(ValueError, "bracketing"):
            m.interpolate_tf(samples, np.array([100, 200]), 201, 100)

    def test_invalid_quaternion_and_invalid_xyz_are_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError, "quaternion"):
            m.quat_matrix([0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "quaternion"):
            m.quat_matrix([float("nan"), 0, 0, 1])
        data = np.array([np.nan, 1, 2], dtype="<f4").tobytes()
        fields = [
            NS(name=n, offset=o, datatype=7, count=1) for n, o in zip(("x", "y", "z"), (0, 4, 8))
        ]
        msg = NS(
            width=1,
            height=1,
            point_step=12,
            row_step=12,
            data=data,
            is_bigendian=False,
            fields=fields,
        )
        self.assertTrue(np.isnan(m.cloud_xyz(msg)[0, 0]))


if __name__ == "__main__":
    unittest.main()
