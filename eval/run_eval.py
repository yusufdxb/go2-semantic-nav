#!/usr/bin/env python3
"""Run the language-grounding evaluation end-to-end.

Two modes:
  - rosbag   : play back a recorded rosbag while semantic-nav consumes it
  - synthetic: use the synthetic publisher (scripts/synthetic_publisher.py) for
               a hardware-free smoke run (useful in CI or Phase-1 validation)

Either way, the harness:
  1. Starts the fixture publisher (rosbag or synthetic) in a subprocess.
  2. Spins up the detector / scene_graph / grounding nodes in-process.
  3. For each query in the YAML, waits for the scene graph to warm up, fires
     the `/semantic/ground_and_navigate` action (dry-run), and records:
       - chosen label, top-5 labels, grounding score, grounding latency.
  4. Writes results.csv + summary.json and prints aggregate metrics.

This intentionally does NOT drive a robot or measure navigation success; SR/SPL
require hardware or Gazebo and will be filled in when hardware is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required (pip install pyyaml)", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).parent))
from metrics import QueryOutcome, summarize  # noqa: E402


def _spawn_fixture(mode: str, bag_path: Optional[str], publisher_script: Path) -> Optional[subprocess.Popen]:
    if mode == "synthetic":
        print(f"[eval] launching synthetic publisher: {publisher_script}")
        return subprocess.Popen(
            [sys.executable, str(publisher_script),
             "--ros-args", "-p", "rate_hz:=5.0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    if mode == "rosbag":
        if not bag_path:
            raise SystemExit("--bag is required when mode=rosbag")
        print(f"[eval] launching rosbag play: {bag_path}")
        return subprocess.Popen(
            ["ros2", "bag", "play", bag_path, "--loop"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    raise SystemExit(f"unknown mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", default="eval/queries.yaml")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "rosbag"])
    ap.add_argument("--bag", default=None)
    ap.add_argument("--out", default="eval/results/latest")
    ap.add_argument("--config-name", default="dev_default")
    ap.add_argument("--warmup-s", type=float, default=20.0)
    ap.add_argument("--per-query-timeout-s", type=float, default=20.0)
    ap.add_argument("--detector", default="yolo_world_v2_s")
    ap.add_argument("--segmenter", default="mobile_sam")
    ap.add_argument("--encoder", default="openclip_vit_b16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.queries) as f:
        cfg = yaml.safe_load(f)
    queries = cfg.get("queries", [])
    if not queries:
        print("no queries found", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    pub_script = repo_root / "scripts" / "synthetic_publisher.py"
    pub_proc = _spawn_fixture(args.mode, args.bag, pub_script)

    try:
        import rclpy
        import rclpy.parameter as p_mod
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor

        from go2_language_grounding.grounding_node import GroundingNode
        from go2_open_vocab_detector.detector_node import DetectorNode
        from go2_scene_graph.scene_graph_node import SceneGraphNode
        from go2_semantic_msgs.action import GroundAndNavigate

        rclpy.init()

        detector = DetectorNode()
        detector.set_parameters([
            p_mod.Parameter("detector_backend", p_mod.Parameter.Type.STRING, args.detector),
            p_mod.Parameter("segmenter_backend", p_mod.Parameter.Type.STRING, args.segmenter),
            p_mod.Parameter("encoder_backend", p_mod.Parameter.Type.STRING, args.encoder),
            p_mod.Parameter("device", p_mod.Parameter.Type.STRING, args.device),
        ])
        scene_graph = SceneGraphNode()
        scene_graph.set_parameters([
            p_mod.Parameter("min_observations_to_publish", p_mod.Parameter.Type.INTEGER, 1),
            p_mod.Parameter("assoc_embed_threshold", p_mod.Parameter.Type.DOUBLE, 0.5),
            p_mod.Parameter("publish_rate_hz", p_mod.Parameter.Type.DOUBLE, 4.0),
        ])
        grounding = GroundingNode()
        grounding.set_parameters([
            p_mod.Parameter("encoder_backend", p_mod.Parameter.Type.STRING, args.encoder),
            p_mod.Parameter("device", p_mod.Parameter.Type.STRING, args.device),
            p_mod.Parameter("use_costmap_gate", p_mod.Parameter.Type.BOOL, False),
        ])

        executor = MultiThreadedExecutor()
        for n in (detector, scene_graph, grounding):
            executor.add_node(n)

        print(f"[eval] warmup {args.warmup_s:.0f} s for scene graph to populate ...")
        start = time.time()
        while time.time() - start < args.warmup_s:
            executor.spin_once(timeout_sec=0.1)

        action = ActionClient(grounding, GroundAndNavigate, "/semantic/ground_and_navigate")
        if not action.wait_for_server(timeout_sec=30.0):
            raise SystemExit("grounding action server not available after warmup")

        outcomes: list[QueryOutcome] = []
        for q in queries:
            qid = str(q.get("id", ""))
            text = str(q.get("query", ""))
            scene = str(q.get("scene", ""))
            expected = q.get("expected_label")

            goal = GroundAndNavigate.Goal()
            goal.text_query = text
            goal.stand_off_m = 0.9
            goal.timeout_seconds = float(args.per_query_timeout_s)
            goal.dry_run = args.dry_run
            goal.top_k = 5

            t0 = time.time()
            send_fut = action.send_goal_async(goal)
            while not send_fut.done() and time.time() - t0 < args.per_query_timeout_s:
                executor.spin_once(timeout_sec=0.1)

            if not send_fut.done() or send_fut.result() is None or not send_fut.result().accepted:
                outcomes.append(QueryOutcome(
                    query_id=qid, query=text, scene=scene, expected_label=expected,
                    success=False, chosen_label=None, top5_labels=[],
                    grounding_score=0.0, grounding_latency_ms=(time.time() - t0) * 1000,
                ))
                print(f"[eval] {qid}: REJECTED/TIMEOUT")
                continue

            result_fut = send_fut.result().get_result_async()
            while not result_fut.done() and time.time() - t0 < args.per_query_timeout_s:
                executor.spin_once(timeout_sec=0.1)

            lat_ms = (time.time() - t0) * 1000
            if not result_fut.done():
                outcomes.append(QueryOutcome(
                    query_id=qid, query=text, scene=scene, expected_label=expected,
                    success=False, chosen_label=None, top5_labels=[],
                    grounding_score=0.0, grounding_latency_ms=lat_ms,
                ))
                print(f"[eval] {qid}: action result timeout")
                continue

            r = result_fut.result().result
            outcomes.append(QueryOutcome(
                query_id=qid, query=text, scene=scene, expected_label=expected,
                success=bool(r.success),
                chosen_label=r.chosen_object_label or None,
                top5_labels=[],
                grounding_score=float(r.grounding_score),
                grounding_latency_ms=lat_ms,
            ))
            print(
                f"[eval] {qid}: success={r.success} chose={r.chosen_object_label!r} "
                f"score={r.grounding_score:.3f} lat={lat_ms:.0f}ms"
            )

        # Write results
        results_csv = out_dir / "results.csv"
        if outcomes:
            fieldnames = list(asdict(outcomes[0]).keys())
            with results_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for o in outcomes:
                    row = asdict(o)
                    row["top5_labels"] = "|".join(o.top5_labels)
                    w.writerow(row)

        summary = summarize(outcomes)
        summary["config_name"] = args.config_name
        summary["mode"] = args.mode
        summary["n_queries"] = len(outcomes)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print("\n=== eval summary ===")
        for k, v in summary.items():
            print(f"  {k:30s}: {v}")
        print(f"[eval] wrote {results_csv} and summary.json to {out_dir}")
        return 0
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if pub_proc is not None:
            pub_proc.terminate()
            try:
                pub_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pub_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
