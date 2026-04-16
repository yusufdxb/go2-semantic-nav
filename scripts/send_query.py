#!/usr/bin/env python3
"""Convenience wrapper around /semantic/ground_and_navigate.

Replaces the verbose `ros2 action send_goal` incantation with a one-liner:
    python scripts/send_query.py "go near the red chair"
    python scripts/send_query.py --commit "go near the red chair"   # publishes /goal_pose
    python scripts/send_query.py --stand-off 0.6 "approach the window"
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from go2_semantic_msgs.action import GroundAndNavigate
from rclpy.action import ActionClient
from rclpy.node import Node


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", help="Natural-language target description")
    ap.add_argument("--stand-off", type=float, default=0.9, dest="stand_off")
    ap.add_argument("--commit", action="store_true", help="Publish /goal_pose (default: dry-run)")
    ap.add_argument("--relation", default="", help="Override parsed relation (e.g., near, left_of)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("send_query_cli")
    client = ActionClient(node, GroundAndNavigate, "/semantic/ground_and_navigate")
    if not client.wait_for_server(timeout_sec=5.0):
        print("ERROR: grounding action server not available", file=sys.stderr)
        rclpy.shutdown()
        return 1

    goal = GroundAndNavigate.Goal()
    goal.text_query = args.query
    goal.stand_off_m = float(args.stand_off)
    goal.timeout_seconds = float(args.timeout)
    goal.dry_run = not args.commit
    goal.top_k = int(args.top_k)
    goal.preferred_relation = args.relation

    print(f"[send_query] dispatching: {args.query!r}  stand_off={args.stand_off}m "
          f"commit={args.commit} relation={args.relation or '(auto-parse)'}")

    send_fut = client.send_goal_async(goal)
    t0 = time.time()
    while not send_fut.done() and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not send_fut.done():
        print("ERROR: send_goal timed out", file=sys.stderr)
        rclpy.shutdown()
        return 2

    goal_handle = send_fut.result()
    if not goal_handle.accepted:
        print("REJECTED: server refused the goal (scene graph empty?)", file=sys.stderr)
        rclpy.shutdown()
        return 3

    result_fut = goal_handle.get_result_async()
    while not result_fut.done() and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not result_fut.done():
        print("ERROR: result timed out", file=sys.stderr)
        rclpy.shutdown()
        return 4

    r = result_fut.result().result
    print(f"state=         {r.terminal_state}")
    print(f"success=       {r.success}")
    print(f"message=       {r.message}")
    print(f"chosen_label=  {r.chosen_object_label!r}")
    print(f"chosen_id=     {r.chosen_object_id}")
    print(f"score=         {r.grounding_score:.3f}")
    print(f"final_goal=    frame={r.final_goal.header.frame_id} "
          f"xyz=({r.final_goal.pose.position.x:.3f}, "
          f"{r.final_goal.pose.position.y:.3f}, "
          f"{r.final_goal.pose.position.z:.3f})")
    print(f"latency=       {r.total_latency_s * 1000:.1f} ms")

    rclpy.shutdown()
    return 0 if r.success else 10


if __name__ == "__main__":
    raise SystemExit(main())
