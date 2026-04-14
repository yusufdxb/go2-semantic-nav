#!/usr/bin/env python3
"""End-to-end smoke test for go2-semantic-nav without real hardware or rosbags.

Flow:
  1. Start the synthetic RGB-D publisher (scripts/synthetic_publisher.py)
  2. Start the detector, scene_graph, and grounding nodes in-process
  3. Wait for /semantic/scene_graph to contain ≥1 object
  4. Send a GroundAndNavigate action with text_query="person" (dry_run=true)
  5. Assert success + PoseStamped in map frame

Exits 0 on success, non-zero on failure. Designed for Phase 1 validation.

Run from the repo root with the workspace sourced:
    source ros2_ws/install/setup.bash
    python3 scripts/smoke_test_integration.py --text "person" --timeout 90
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node

from go2_semantic_msgs.action import GroundAndNavigate
from go2_semantic_msgs.msg import SceneGraph


class SceneGraphWatcher(Node):
    """Listens for /semantic/scene_graph and exposes the latest message."""

    def __init__(self) -> None:
        super().__init__("smoke_scene_graph_watcher")
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        q = QoSProfile(depth=5)
        q.reliability = ReliabilityPolicy.RELIABLE
        self.latest: Optional[SceneGraph] = None
        self._sub = self.create_subscription(SceneGraph, "/semantic/scene_graph", self._on, q)

    def _on(self, msg: SceneGraph) -> None:
        self.latest = msg


def _wait_for_scene_graph(watcher: SceneGraphWatcher, executor, timeout_s: float) -> Optional[SceneGraph]:
    start = time.time()
    while time.time() - start < timeout_s:
        executor.spin_once(timeout_sec=0.1)
        if watcher.latest is not None and len(watcher.latest.nodes) > 0:
            return watcher.latest
    return watcher.latest  # may be empty or None


def _send_grounding_goal(node: Node, executor, text: str, timeout_s: float, dry_run: bool) -> GroundAndNavigate.Result:
    client = ActionClient(node, GroundAndNavigate, "/semantic/ground_and_navigate")
    if not client.wait_for_server(timeout_sec=timeout_s):
        raise RuntimeError("grounding action server not available within timeout")

    goal = GroundAndNavigate.Goal()
    goal.text_query = text
    goal.stand_off_m = 0.9
    goal.timeout_seconds = float(timeout_s)
    goal.dry_run = dry_run
    goal.top_k = 5

    send_future = client.send_goal_async(goal)
    start = time.time()
    while not send_future.done() and time.time() - start < timeout_s:
        executor.spin_once(timeout_sec=0.1)
    if not send_future.done():
        raise RuntimeError("send_goal_async timed out")
    goal_handle = send_future.result()
    if not goal_handle.accepted:
        raise RuntimeError("goal was rejected by server")

    result_future = goal_handle.get_result_async()
    start = time.time()
    while not result_future.done() and time.time() - start < timeout_s:
        executor.spin_once(timeout_sec=0.1)
    if not result_future.done():
        raise RuntimeError("result_future timed out")
    return result_future.result().result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default="person", help="text query for grounding action")
    ap.add_argument("--timeout", type=float, default=120.0, help="overall timeout seconds")
    ap.add_argument("--publisher-rate", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--encoder", default="openclip_vit_b16")
    ap.add_argument("--detector", default="yolo_world_v2_s")
    ap.add_argument("--segmenter", default="mobile_sam")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--assume-publisher-running", action="store_true",
                    help="skip launching synthetic_publisher.py (useful if it's already up)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # 1. Launch synthetic publisher + static TF in a subprocess.
    pub_proc: Optional[subprocess.Popen] = None
    if not args.assume_publisher_running:
        pub_script = str(repo_root / "scripts" / "synthetic_publisher.py")
        print(f"[smoke] launching {pub_script}")
        pub_proc = subprocess.Popen(
            [sys.executable, pub_script,
             "--ros-args", "-p", f"rate_hz:={args.publisher_rate}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    try:
        rclpy.init()

        # 2. Construct the three semantic-nav nodes in-process.
        print("[smoke] loading detector (this will pull weights on first run) ...")
        from go2_open_vocab_detector.detector_node import DetectorNode
        from go2_scene_graph.scene_graph_node import SceneGraphNode
        from go2_language_grounding.grounding_node import GroundingNode
        import rclpy.parameter as p_mod

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

        watcher = SceneGraphWatcher()

        executor = MultiThreadedExecutor()
        for n in (detector, scene_graph, grounding, watcher):
            executor.add_node(n)

        print("[smoke] waiting up to 60 s for scene graph to populate ...")
        sg = _wait_for_scene_graph(watcher, executor, timeout_s=60.0)
        if sg is None:
            print("[smoke] FAIL: no SceneGraph message received in 60 s")
            return 2
        if len(sg.nodes) == 0:
            print("[smoke] FAIL: SceneGraph message arrived but contained zero objects")
            print(f"[smoke]   header.frame_id={sg.header.frame_id}")
            return 2
        labels = [n.label for n in sg.nodes]
        print(f"[smoke] scene graph populated: {len(sg.nodes)} objects, labels={labels}")

        print(f"[smoke] dispatching grounding action with text={args.text!r}")
        try:
            result = _send_grounding_goal(
                grounding, executor,
                text=args.text, timeout_s=30.0, dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"[smoke] FAIL: action dispatch errored: {exc}")
            return 3

        print(f"[smoke] action result: success={result.success} state={result.terminal_state}")
        print(f"[smoke]   chosen_object_label={result.chosen_object_label!r}")
        print(f"[smoke]   grounding_score={result.grounding_score:.3f}")
        print(f"[smoke]   final_goal: frame={result.final_goal.header.frame_id} "
              f"x={result.final_goal.pose.position.x:.3f} "
              f"y={result.final_goal.pose.position.y:.3f}")

        # Pass criteria: result.success AND final_goal frame is "map" AND pose is finite
        pos = result.final_goal.pose.position
        pos_finite = all(math.isfinite(v) for v in (pos.x, pos.y, pos.z))
        if not result.success:
            print(f"[smoke] FAIL: action returned success=false: {result.message}")
            return 4
        if result.final_goal.header.frame_id != "map":
            print(f"[smoke] FAIL: final_goal not in map frame: {result.final_goal.header.frame_id}")
            return 5
        if not pos_finite:
            print("[smoke] FAIL: final_goal has non-finite position")
            return 6

        print("[smoke] PASS")
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
