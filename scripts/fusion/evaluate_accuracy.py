#!/usr/bin/env python3
"""Compare recorded fusion surface positions with independently surveyed references.

References are matched by exact image timestamp and independently annotated
object index, never by nearest estimated position. See README.md for schemas.
Exit 0: PASS, 2: FAIL, 3: BLOCKED by missing evidence, 1: malformed input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros2_ws/src/go2_rgb_lidar"))
from go2_rgb_lidar.calibration import camera_model_from_dict, extrinsics_from_dict


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite metre values")
    return result


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_checks(evidence, root):
    """Check documentary prerequisites, without claiming to certify measurements."""
    blockers, hashes = [], {}
    for key in ("timing_validated", "cloud_geometry_validated", "ground_truth_independent"):
        if evidence.get(key) is not True:
            blockers.append(f"{key} is not attested")
    if evidence.get("ground_truth_used_for_calibration") is not False:
        blockers.append("ground truth must be held out from calibration fitting")
    if not isinstance(evidence.get("calibration_id"), str) or not evidence["calibration_id"].strip():
        blockers.append("missing calibration_id")
    for key in ("intrinsics_file", "extrinsics_file", "timing_evidence_file",
                "cloud_geometry_evidence_file", "ground_truth_evidence_file"):
        name = evidence.get(key)
        if not isinstance(name, str) or not name.strip() or not (root / name).is_file():
            blockers.append(f"missing {key}")
            continue
        path = root / name
        hashes[key] = file_hash(path)
        if key in ("intrinsics_file", "extrinsics_file"):
            calibration = yaml.safe_load(path.read_text())
            if not isinstance(calibration, dict) or calibration.get("nominal") is not False:
                blockers.append(f"{key} does not explicitly declare nominal: false")
            elif key == "intrinsics_file":
                cam = camera_model_from_dict(calibration)
                if not (np.isfinite(cam.k).all() and np.isfinite(cam.d).all() and
                        np.allclose(cam.k[2], [0, 0, 1]) and
                        cam.k[0, 1] == 0 and cam.k[1, 0] == 0):
                    raise ValueError("intrinsics must be finite and use supported unskewed K")
            else:
                ext = extrinsics_from_dict(calibration)
                if not np.isfinite([*ext.xyz, *ext.rpy]).all():
                    raise ValueError("extrinsics must be finite")
    return blockers, hashes


def evaluate(predictions, references, evidence, root=Path(".")):
    if evidence.get("data_kind") not in ("measured", "synthetic"):
        raise ValueError("evidence data_kind must be measured or synthetic")
    policy = evidence["acceptance"]
    min_samples = integer(policy["min_valid_samples"], "min_valid_samples")
    if min_samples < 1:
        raise ValueError("min_valid_samples must be positive")
    min_rate = float(policy["min_valid_fraction"])
    p95_limit, max_limit = float(policy["max_p95_error_m"]), float(policy["max_error_m"])
    if not (math.isfinite(min_rate) and 0 < min_rate <= 1 and
            math.isfinite(p95_limit) and p95_limit > 0 and
            math.isfinite(max_limit) and max_limit >= p95_limit):
        raise ValueError("invalid acceptance policy")
    blockers, hashes = evidence_checks(evidence, root)
    frames = {}
    for frame in predictions:
        stamp = integer(frame["stamp_ns"], "stamp_ns")
        if stamp == 0 or stamp in frames:
            raise ValueError("prediction stamps must be positive and unique")
        if not isinstance(frame["objects"], list):
            raise ValueError("objects must be a list")
        objects = {}
        for obj in frame["objects"]:
            index = integer(obj["index"], "object index")
            if index in objects:
                raise ValueError("duplicate prediction object index")
            if type(obj["valid"]) is not bool:
                raise ValueError("prediction valid must be boolean")
            if obj["valid"]:
                vector(obj["position_m"], "prediction position")
                if obj["status"] != 0 or frame["array_status"] != 0:
                    raise ValueError("valid position has invalid object or array status")
            objects[index] = obj
        frames[stamp] = (frame, objects)
    if not references:
        raise ValueError("no independently scheduled references")
    seen, annotated, target_stamps = set(), set(), set()
    errors, upper_errors, residuals, rows = [], [], [], []
    rejected = Counter()
    for ref in references:
        stamp = integer(ref["stamp_ns"], "reference stamp_ns")
        target = str(ref["target_id"]).strip()
        if stamp == 0 or not target or (stamp, target) in target_stamps:
            raise ValueError("reference needs a unique positive stamp and target_id")
        target_stamps.add((stamp, target))
        index = ref["detection_index"]
        if index is not None:
            integer(index, "detection_index")
            if (stamp, index) in seen:
                raise ValueError("reference detection assigned to multiple targets")
            seen.add((stamp, index))
            annotated.add((stamp, index))
        if ref.get("position_definition") != "visible_surface_centroid":
            raise ValueError("reference must measure the same visible surface centroid as fusion")
        if not str(ref.get("measurement_id", "")).strip():
            raise ValueError("reference needs independent measurement_id")
        expected = vector(ref["position_m"], "reference position")
        uncertainty = float(ref["measurement_uncertainty_m"])
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("measurement uncertainty must be a finite nonnegative distance bound")
        row = {"stamp_ns": stamp, "target_id": target, "detection_index": index}
        frame, objects = frames.get(stamp, ({}, {}))
        obj = objects.get(index)
        reason = None
        if index is None:
            reason = "detector_miss"
        elif not frame:
            reason = "missing_output"
        elif frame["frame_id"] != ref["frame_id"]:
            raise ValueError("reference and prediction coordinate frames differ")
        elif frame["calibration_id"] != evidence.get("calibration_id"):
            reason = "calibration_id_mismatch"
            blockers.append("prediction calibration_id differs from evidence")
        elif not obj:
            reason = "missing_detection"
        elif obj["label"] != ref["label"]:
            reason = "label_mismatch"
        elif not obj["valid"]:
            reason = f"invalid_object_status_{obj['status']}"
        if reason:
            rejected[reason] += 1
            row["status"] = reason
        else:
            delta = vector(obj["position_m"], "prediction position") - expected
            error = float(np.linalg.norm(delta))
            errors.append(error)
            upper_errors.append(error + uncertainty)
            residuals.append(delta)
            row.update(status="measured", residual_xyz_m=delta.tolist(), error_m=error,
                       error_upper_bound_m=error + uncertainty,
                       support_points=obj["support_points"], track_id=obj["track_id"])
        rows.append(row)
    n = len(errors)
    fraction = n / len(references)
    metrics = None
    if n:
        metrics = {"bias_xyz_m": np.mean(residuals, axis=0).tolist(),
                   "rmse_3d_m": float(np.sqrt(np.mean(np.square(errors)))),
                   "median_error_m": float(np.median(errors)),
                   "p95_error_m": float(np.percentile(errors, 95)),
                   "max_error_m": max(errors),
                   "p95_error_upper_bound_m": float(np.percentile(upper_errors, 95)),
                   "max_error_upper_bound_m": max(upper_errors)}
    failures = []
    if n < min_samples:
        failures.append("insufficient valid samples")
    if fraction < min_rate:
        failures.append("valid fraction below acceptance policy")
    if n and metrics["p95_error_upper_bound_m"] > p95_limit:
        failures.append("P95 error including reference uncertainty exceeds limit")
    if n and metrics["max_error_upper_bound_m"] > max_limit:
        failures.append("maximum error including reference uncertainty exceeds limit")
    verdict = "BLOCKED" if blockers else "FAIL" if failures else "PASS"
    unreferenced = sum((stamp, index) not in annotated for stamp, (_, objs) in frames.items()
                      for index in objs)
    return {"verdict": verdict, "data_kind": evidence["data_kind"],
            "metric_accuracy_validated": verdict == "PASS" and evidence["data_kind"] == "measured",
            "scope": "recorded visible-surface position on independently annotated targets; no navigation or live-hardware claim",
            "evidence_boundary": "documentary attestations and file hashes are checked; physical survey and calibration quality require independent review",
            "calibration_id": evidence.get("calibration_id"), "evidence_sha256": hashes,
            "acceptance": policy, "reference_count": len(references), "valid_count": n,
            "valid_fraction": fraction, "rejections": dict(rejected),
            "unreferenced_predictions": unreferenced,
            "unreferenced_note": "not scored as false positives; this is targeted localization evaluation",
            "metrics": metrics, "blockers": sorted(set(blockers)), "failures": failures,
            "observations": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "references", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence.read_text())
    root = args.evidence.resolve().parent
    protected = {args.predictions.resolve(), args.references.resolve(), args.evidence.resolve()}
    protected.update((root / value).resolve() for key, value in evidence.items()
                     if key.endswith("_file") and isinstance(value, str))
    if args.output.resolve() in protected:
        raise ValueError("output must not overwrite an input")
    report = evaluate(read_jsonl(args.predictions), read_jsonl(args.references),
                      evidence, root)
    report["input_sha256"] = {key: file_hash(getattr(args, key))
                              for key in ("predictions", "references", "evidence")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("verdict", "reference_count", "valid_count", "valid_fraction", "blockers", "failures")}))
    return {"PASS": 0, "FAIL": 2, "BLOCKED": 3}[report["verdict"]]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
