"""Accuracy gates with analytic errors, missing detections and nominal evidence."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

spec = importlib.util.spec_from_file_location("accuracy", Path(__file__).with_name("evaluate_accuracy.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.evidence = {"data_kind": "synthetic", "calibration_id": "synthetic-test", "timing_validated": True,
                         "cloud_geometry_validated": True, "ground_truth_independent": True,
                         "ground_truth_used_for_calibration": False,
                         "acceptance": {"min_valid_samples": 2, "min_valid_fraction": 1.0,
                                        "max_p95_error_m": 0.15, "max_error_m": 0.15}}
        for key in ("intrinsics_file", "extrinsics_file", "timing_evidence_file",
                    "cloud_geometry_evidence_file", "ground_truth_evidence_file"):
            self.evidence[key] = key + ".yaml"
            (self.root / self.evidence[key]).write_text("nominal: false\n")
        (self.root / self.evidence["intrinsics_file"]).write_text(yaml.safe_dump({
            "nominal": False, "image_width": 640, "image_height": 360,
            "camera_matrix": {"data": [320, 0, 319.5, 0, 320, 179.5, 0, 0, 1]},
            "distortion_model": "plumb_bob", "distortion_coefficients": {"data": [0] * 5}}))
        (self.root / self.evidence["extrinsics_file"]).write_text(yaml.safe_dump({
            "nominal": False, "parent_frame": "base_link", "child_frame": "camera",
            "xyz": [0, 0, 0], "rpy": [0, 0, 0]}))
        self.frames, self.refs = [], []
        for stamp, point in [(1, [1.03, 2.04, 3]), (2, [1, 2, 3.12])]:
            self.frames.append({"stamp_ns": stamp, "frame_id": "odom", "array_status": 0,
                                "calibration_id": "synthetic-test", "objects": [
                                    {"index": 0, "label": "chair", "status": 0, "valid": True,
                                     "position_m": point, "track_id": 1, "support_points": 9}]})
            self.refs.append({"stamp_ns": stamp, "frame_id": "odom", "target_id": "chair_a",
                              "label": "chair", "detection_index": 0, "position_m": [1, 2, 3],
                              "position_definition": "visible_surface_centroid",
                              "measurement_id": "independent-survey", "measurement_uncertainty_m": 0.01})

    def run_eval(self):
        return m.evaluate(self.frames, self.refs, self.evidence, self.root)

    def test_analytic_xyz_errors_and_conservative_reference_uncertainty(self):
        report = self.run_eval()
        self.assertEqual(report["verdict"], "PASS")
        self.assertFalse(report["metric_accuracy_validated"])
        np.testing.assert_allclose(report["metrics"]["bias_xyz_m"], [.015, .02, .06])
        self.assertAlmostEqual(report["metrics"]["rmse_3d_m"], np.sqrt((.05**2 + .12**2) / 2))
        self.assertAlmostEqual(report["metrics"]["max_error_upper_bound_m"], .13)
        self.refs[1]["measurement_uncertainty_m"] = .05
        self.assertEqual(self.run_eval()["verdict"], "FAIL")

    def test_missing_invalid_and_detector_miss_remain_in_denominator(self):
        for mode in ("missing", "invalid", "miss"):
            with self.subTest(mode=mode):
                frames, refs = copy.deepcopy(self.frames), copy.deepcopy(self.refs)
                if mode == "missing":
                    frames.pop()
                elif mode == "invalid":
                    frames[1]["objects"][0].update(valid=False, status=6, position_m=None)
                else:
                    refs[1]["detection_index"] = None
                report = m.evaluate(frames, refs, self.evidence, self.root)
                self.assertEqual(report["verdict"], "FAIL")
                self.assertEqual(report["reference_count"], 2)
                self.assertEqual(report["valid_fraction"], .5)

    def test_nominal_missing_or_fitted_truth_blocks_an_accuracy_claim(self):
        for mode in ("nominal", "missing", "fitted"):
            with self.subTest(mode=mode):
                evidence = copy.deepcopy(self.evidence)
                if mode == "nominal":
                    (self.root / "nominal.yaml").write_text("nominal: true\n")
                    evidence["intrinsics_file"] = "nominal.yaml"
                elif mode == "missing":
                    evidence["timing_evidence_file"] = "absent.txt"
                else:
                    evidence["ground_truth_used_for_calibration"] = True
                report = m.evaluate(self.frames, self.refs, evidence, self.root)
                self.assertEqual(report["verdict"], "BLOCKED")
                self.assertFalse(report["metric_accuracy_validated"])

    def test_matching_never_uses_closest_prediction(self):
        self.frames[0]["objects"].append(dict(self.frames[0]["objects"][0], index=1, position_m=[1, 2, 3]))
        report = self.run_eval()
        self.assertAlmostEqual(report["observations"][0]["error_m"], .05)
        self.assertEqual(report["unreferenced_predictions"], 1)
        self.frames[0]["objects"][0]["label"] = "table"
        self.assertEqual(self.run_eval()["verdict"], "FAIL")

    def test_wrong_frames_nonfinite_valid_and_definition_are_rejected(self):
        for mode in ("frame", "nonfinite", "definition", "duplicate"):
            with self.subTest(mode=mode):
                frames, refs = copy.deepcopy(self.frames), copy.deepcopy(self.refs)
                if mode == "frame":
                    refs[0]["frame_id"] = "base_link"
                elif mode == "nonfinite":
                    frames[0]["objects"][0]["position_m"][0] = float("nan")
                elif mode == "definition":
                    refs[0]["position_definition"] = "chair_center"
                else:
                    refs.append(copy.deepcopy(refs[0]))
                with self.assertRaises(ValueError):
                    m.evaluate(frames, refs, self.evidence, self.root)

    def test_calibration_mismatch_and_empty_reference_cannot_pass(self):
        self.frames[0]["calibration_id"] = "different"
        self.assertEqual(self.run_eval()["verdict"], "BLOCKED")
        with self.assertRaises(ValueError):
            m.evaluate(self.frames, [], self.evidence, self.root)


if __name__ == "__main__":
    unittest.main()
