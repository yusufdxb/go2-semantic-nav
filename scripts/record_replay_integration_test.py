#!/usr/bin/env python3
"""Record-and-replay integration test.

Proves the rosbag pathway that Phase 5 relies on, without needing robot hardware:

  1. Start the synthetic publisher.
  2. `ros2 bag record` the required topic set for ~N seconds.
  3. Stop the publisher.
  4. `ros2 bag play` the recorded bag in a loop.
  5. Run the 3 semantic-nav nodes + grounding action.
  6. Assert the grounding action returns success=True on a known-to-work query.

**Known limitation (documented 2026-04-13):** ros2 bag play replays messages with
their ORIGINAL timestamps (from the recording). ROS 2 subscribers consume them
against wall-clock time by default, so TF lookups in `scene_graph_node` fail
because the requested `stamp` predates the TF cache window. Two-process fix:
pass `--clock` to `ros2 bag play` AND set `use_sim_time:=true` on all subscriber
nodes. A cleaner Phase 5 path is to record from the real robot (where `/tf` is
live) and replay with `ros2 bag play --clock --rate 1.0` + `use_sim_time:=true`.

For hardware-free CI validation, use `scripts/smoke_test_integration.py` — it
uses the live synthetic publisher (not a bag) so wall-clock TFs work fine.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _run_and_wait(cmd: list[str], timeout_s: float, name: str) -> tuple[int, str, str]:
    """Run a command synchronously and capture output."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\n[{name}] timed out after {timeout_s}s"


def _record_bag(record_duration_s: float, bag_path: Path, publisher_rate_hz: float) -> Path:
    """Start publisher, record a bag for `record_duration_s`, stop publisher."""
    repo_root = Path(__file__).resolve().parents[1]
    pub_script = repo_root / "scripts" / "synthetic_publisher.py"

    if bag_path.exists():
        shutil.rmtree(bag_path)

    print(f"[record] publisher: {pub_script}")
    pub = subprocess.Popen(
        [sys.executable, str(pub_script),
         "--ros-args", "-p", f"rate_hz:={publisher_rate_hz}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # let publisher come up

    topics = [
        "/camera/color/image_raw",
        "/camera/depth/image_rect_raw",
        "/camera/color/camera_info",
        "/tf_static",
    ]
    # QoS override — BEST_EFFORT image publishers drop frames if the recorder
    # subscribes with default RELIABLE. We saw 0.9 Hz captured from an 8 Hz
    # publisher without this override.
    qos_path = bag_path.parent / f"{bag_path.name}_qos_override.yaml"
    qos_path.write_text(
        "/camera/color/image_raw:\n"
        "  reliability: best_effort\n  history: keep_last\n  depth: 1\n"
        "/camera/depth/image_rect_raw:\n"
        "  reliability: best_effort\n  history: keep_last\n  depth: 1\n"
    )
    print(f"[record] recording {len(topics)} topics for {record_duration_s} s → {bag_path}")
    rec = subprocess.Popen(
        ["ros2", "bag", "record",
         "-o", str(bag_path),
         "--qos-profile-overrides-path", str(qos_path)] + topics,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(record_duration_s)
    rec.terminate()
    try:
        rec.wait(timeout=10)
    except subprocess.TimeoutExpired:
        rec.kill()
    pub.terminate()
    try:
        pub.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pub.kill()
    if not bag_path.exists():
        raise RuntimeError(f"bag not written at {bag_path}")
    print(f"[record] wrote {bag_path}")
    return bag_path


def _replay_and_ground(bag_path: Path, text_query: str, timeout_s: float) -> dict:
    """Play back the bag in a loop and dispatch a grounding action."""
    # `--clock` tells rosbag play to also publish /clock so sim-time subscribers can
    # align their TF lookups. Subscriber nodes must be launched with use_sim_time:=true
    # for this to close the loop; this test constructs nodes directly so that
    # remains a known limitation (see module docstring).
    play = subprocess.Popen(
        ["ros2", "bag", "play", str(bag_path), "--loop", "--clock"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2)  # initial buffering

        import rclpy
        import rclpy.parameter as p_mod
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor

        from go2_language_grounding.grounding_node import GroundingNode
        from go2_open_vocab_detector.detector_node import DetectorNode
        from go2_scene_graph.scene_graph_node import SceneGraphNode
        from go2_semantic_msgs.action import GroundAndNavigate

        # Pass use_sim_time via rclpy.init so every node inherits it from the context.
        rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
        det = DetectorNode()
        det.set_parameters([
            p_mod.Parameter("use_sim_time", p_mod.Parameter.Type.BOOL, True),
            p_mod.Parameter("device", p_mod.Parameter.Type.STRING, "cuda:0"),
        ])
        sg = SceneGraphNode()
        sg.set_parameters([
            p_mod.Parameter("use_sim_time", p_mod.Parameter.Type.BOOL, True),
            p_mod.Parameter("min_observations_to_publish", p_mod.Parameter.Type.INTEGER, 1),
            p_mod.Parameter("assoc_embed_threshold", p_mod.Parameter.Type.DOUBLE, 0.5),
            p_mod.Parameter("publish_rate_hz", p_mod.Parameter.Type.DOUBLE, 4.0),
            # TF cache needs to be permissive when /clock is catching up on bag start.
            p_mod.Parameter("tf_lookup_timeout_s", p_mod.Parameter.Type.DOUBLE, 0.5),
        ])
        gr = GroundingNode()
        gr.set_parameters([
            p_mod.Parameter("use_sim_time", p_mod.Parameter.Type.BOOL, True),
            p_mod.Parameter("use_costmap_gate", p_mod.Parameter.Type.BOOL, False),
        ])

        executor = MultiThreadedExecutor()
        for n in (det, sg, gr):
            executor.add_node(n)

        t0 = time.time()
        # Wait for scene graph to populate AND be received by the grounding node's
        # subscriber cache (the grounding action's accept-handler checks that cache).
        print(f"[replay] warmup up to 30 s for scene graph ...")
        while time.time() - t0 < 30.0:
            executor.spin_once(timeout_sec=0.2)
            internal_ok = len(sg._store.objects_with_min_observations()) > 0
            cached = gr._latest_scene_graph
            published_ok = cached is not None and len(cached.nodes) > 0
            if internal_ok and published_ok:
                print(f"[replay] scene graph populated AND published after {time.time() - t0:.1f} s")
                break

        # Dispatch grounding action
        print(f"[replay] dispatching grounding action: {text_query!r}")
        client = ActionClient(gr, GroundAndNavigate, "/semantic/ground_and_navigate")
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("action server not available")

        goal = GroundAndNavigate.Goal()
        goal.text_query = text_query
        goal.stand_off_m = 0.9
        goal.timeout_seconds = float(timeout_s)
        goal.dry_run = True
        goal.top_k = 5

        send_fut = client.send_goal_async(goal)
        t1 = time.time()
        while not send_fut.done() and time.time() - t1 < timeout_s:
            executor.spin_once(timeout_sec=0.1)
        if not send_fut.done() or not send_fut.result().accepted:
            raise RuntimeError("goal rejected or timeout")

        result_fut = send_fut.result().get_result_async()
        while not result_fut.done() and time.time() - t1 < timeout_s:
            executor.spin_once(timeout_sec=0.1)
        if not result_fut.done():
            raise RuntimeError("result timeout")

        r = result_fut.result().result
        out = {
            "success": bool(r.success),
            "terminal_state": r.terminal_state,
            "chosen_label": r.chosen_object_label,
            "grounding_score": float(r.grounding_score),
            "final_goal_frame": r.final_goal.header.frame_id,
            "final_goal_xyz": (
                float(r.final_goal.pose.position.x),
                float(r.final_goal.pose.position.y),
                float(r.final_goal.pose.position.z),
            ),
        }
        return out
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        play.terminate()
        try:
            play.wait(timeout=3)
        except subprocess.TimeoutExpired:
            play.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-duration-s", type=float, default=8.0)
    ap.add_argument("--publisher-rate-hz", type=float, default=6.0)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--text-query", default="person")
    ap.add_argument("--bag-out", default="tests/bags/synthetic_bus")
    ap.add_argument("--skip-record", action="store_true",
                    help="Reuse an existing bag (debug-only)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    bag_path = repo_root / args.bag_out
    bag_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_record or not bag_path.exists():
        _record_bag(args.record_duration_s, bag_path, args.publisher_rate_hz)
    else:
        print(f"[record] reusing existing bag at {bag_path}")

    out = _replay_and_ground(bag_path, args.text_query, args.timeout_s)
    print("\n=== replay result ===")
    for k, v in out.items():
        print(f"  {k:20s} {v}")

    # Pass criteria
    if not out["success"]:
        print("[test] FAIL: action returned success=false")
        return 2
    if out["final_goal_frame"] != "map":
        print(f"[test] FAIL: goal frame {out['final_goal_frame']!r} != 'map'")
        return 3
    x, y, z = out["final_goal_xyz"]
    if not all(math.isfinite(v) for v in (x, y, z)):
        print("[test] FAIL: non-finite goal coords")
        return 4

    print("[test] PASS — record → replay → ground produced a valid PoseStamped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
