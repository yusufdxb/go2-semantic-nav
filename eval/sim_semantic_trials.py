#!/usr/bin/env python3
"""
sim_semantic_trials.py: language query -> motion, closed loop, in simulation.

    python3 eval/sim_semantic_trials.py --trials 2 --planner nav2 --out /tmp/sem_trials

Requires the GO2-seeing-eye-dog workspace (go2_sim, go2_bringup, ...) and this
workspace to be built and sourced. Per trial, on an isolated ROS domain:

  go2_sim (kinematic GO2, real Sport API surface, simulated L1 LiDAR)
  + go2_semantic_bringup/deploy.launch.py (localization, planner, LiDAR hazard
    source, safety arbiter, hardware bridge with the REAL Unitree adapter,
    grounding with the interlock ON)
  + scripts/sim_scene_graph.py (perception STAND-IN from the world file)

then sends each query as GroundAndNavigate(dry_run=false) and scores it
against ground truth:

  present target: success, correct label, terminal_state REACHED, robot
                  stops between min/max surface distance from the object,
                  zero collisions
  absent target:  rejected (GROUNDING_FAILED) and the robot does not move

Perception is a stand-in here; these numbers are evidence that the grounding
-> navigation -> safety -> actuation chain works, not perception accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

QUERIES = [
    ("go to the chair", "chair"),
    ("take me to the refrigerator", "refrigerator"),
    ("the couch", "couch"),
    ("walk to the table", "table"),
    ("go near the door", "door"),
    ("find the dog", None),
    ("go to the piano", None),
    ("the bicycle", None),
]


def _surface_distance(px: float, py: float, obj: dict) -> float:
    """Distance from a point to an axis-aligned object footprint."""
    hx, hy = obj.get("size_x", 0.5) / 2, obj.get("size_y", 0.5) / 2
    dx = max(abs(px - obj["x"]) - hx, 0.0)
    dy = max(abs(py - obj["y"]) - hy, 0.0)
    return math.hypot(dx, dy)


def _launch(domain: int, planner: str, world: str, log: Path) -> subprocess.Popen:
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY="1")
    cmd = (
        f"ros2 launch go2_sim sim.launch.py world_file:={world} & sleep 2; "
        f"ros2 launch go2_semantic_bringup deploy.launch.py planner:={planner} "
        "localization:=slam_mapping hardware_adapter:=unitree_sport allow_goal_publication:=true "
        "enable_detector:=false enable_scene_graph:=false & "
        f"python3 {REPO}/scripts/sim_scene_graph.py {world} & wait"
    )
    return subprocess.Popen(["bash", "-c", cmd], env=env, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop(proc: subprocess.Popen, grace_s: float = 15.0) -> None:
    """Stop EVERY process of the trial, not just the launch wrapper.

    Waiting on the wrapper alone left Nav2 servers alive after SIGINT; they
    kept answering lifecycle and TF traffic on the same domain and poisoned
    later trials that reused it (observed: planner_server survivors from
    every trial of a batch).
    """
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and _group_alive(pgid):
        time.sleep(0.2)
    if _group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run(objects: dict, query_timeout: float, ready_timeout: float,
         min_surface: float, max_surface: float, queries=None, need_costmap: bool = True) -> dict:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from go2_semantic_msgs.action import GroundAndNavigate
    from nav_msgs.msg import OccupancyGrid
    from rclpy.action import ActionClient
    from std_msgs.msg import Bool, UInt32

    rclpy.init()
    node = rclpy.create_node("sim_semantic_trials")
    st = {"valid": False, "gt": None, "collisions": 0, "costmap": False, "graph": False}
    node.create_subscription(Bool, "/go2/localization_valid", lambda m: st.update(valid=m.data), 10)
    node.create_subscription(PoseStamped, "/go2_sim/ground_truth", lambda m: st.update(gt=m), 10)
    node.create_subscription(UInt32, "/go2_sim/collisions", lambda m: st.update(collisions=m.data), 10)
    node.create_subscription(OccupancyGrid, "/global_costmap/costmap", lambda m: st.update(costmap=True), 1)
    client = ActionClient(node, GroundAndNavigate, "/semantic/ground_and_navigate")

    def spin_until(pred, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not pred():
            rclpy.spin_once(node, timeout_sec=0.05)
        return pred()

    out = {"ready": False, "queries": []}
    out["ready"] = bool(spin_until(
        lambda: st["valid"] and (st["costmap"] or not need_costmap) and st["gt"] is not None
        and client.server_is_ready(),
        ready_timeout))
    if not out["ready"]:
        node.destroy_node()
        rclpy.shutdown()
        return out
    spin_until(lambda: False, 3.0)

    for text, expected in (queries or QUERIES):
        p0 = st["gt"].pose.position
        start_xy = (p0.x, p0.y)
        req = GroundAndNavigate.Goal()
        req.text_query = text
        req.dry_run = False
        req.top_k = 5
        req.timeout_seconds = float(query_timeout)
        t0 = time.monotonic()
        send = client.send_goal_async(req)
        spin_until(send.done, 5.0)
        handle = send.result() if send.done() else None
        rec = {"query": text, "expected": expected}
        if handle is None or not handle.accepted:
            rec.update(accepted=False, success=False, terminal_state="REJECTED_AT_ACCEPT", chosen="")
        else:
            res = handle.get_result_async()
            if not spin_until(res.done, query_timeout + 15.0):
                handle.cancel_goal_async()
                spin_until(lambda: False, 3.0)
                rec.update(accepted=True, success=False, terminal_state="HARNESS_TIMEOUT", chosen="")
            else:
                r = res.result().result
                rec.update(accepted=True, success=bool(r.success), terminal_state=r.terminal_state,
                           chosen=r.chosen_object_label, message=r.message)
        p = st["gt"].pose.position
        moved = math.hypot(p.x - start_xy[0], p.y - start_xy[1])
        rec["seconds"] = round(time.monotonic() - t0, 1)
        rec["moved_m"] = round(moved, 3)
        rec["collisions_total"] = int(st["collisions"])
        if expected is not None:
            surf = _surface_distance(p.x, p.y, objects[expected])
            rec["surface_distance_m"] = round(surf, 3)
            rec["pass"] = bool(rec["success"] and rec["chosen"] == expected
                               and rec["terminal_state"] == "REACHED"
                               and min_surface <= surf <= max_surface)
        else:
            rec["pass"] = bool((not rec["success"]) and moved < 0.05)
        out["queries"].append(rec)
        print(f"  {text!r}: {rec['terminal_state']} chose={rec.get('chosen')!r} "
              f"pass={rec['pass']} moved={rec['moved_m']} surf={rec.get('surface_distance_m')}",
              flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--planner", default="nav2", choices=["nav2", "staged_nav"])
    ap.add_argument("--world", default=None)
    ap.add_argument("--query-timeout", type=float, default=150.0)
    ap.add_argument("--ready-timeout", type=float, default=120.0)
    ap.add_argument("--min-surface", type=float, default=0.5,
                    help="closest acceptable stop to the object surface (m)")
    ap.add_argument("--max-surface", type=float, default=1.5)
    ap.add_argument("--domain-base", type=int, default=170)
    ap.add_argument("--queries", default=None,
                    help="'text=label;text=' (empty label = must be refused); default: apartment set")
    ap.add_argument("--out", default="eval/results/sim_semantic")
    args = ap.parse_args(argv)

    world = args.world or subprocess.check_output(
        ["ros2", "pkg", "prefix", "go2_sim"], text=True).strip() + "/share/go2_sim/worlds/apartment.yaml"
    objects = {o["label"]: o for o in yaml.safe_load(open(world))["objects"]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    for i in range(args.trials):
        domain = args.domain_base + i
        proc = _launch(domain, args.planner, world, out_dir / f"trial_{i}.log")
        prev = os.environ.get("ROS_DOMAIN_ID")
        os.environ["ROS_DOMAIN_ID"], os.environ["ROS_LOCALHOST_ONLY"] = str(domain), "1"
        try:
            print(f"trial {i} (domain {domain}, planner {args.planner})", flush=True)
            queries = None
            if args.queries:
                queries = [(q.split("=")[0], q.split("=")[1] or None) for q in args.queries.split(";")]
            res = _run(objects, args.query_timeout, args.ready_timeout, args.min_surface,
                       args.max_surface, queries, need_costmap=(args.planner == "nav2"))
        finally:
            _stop(proc)
            if prev is None:
                os.environ.pop("ROS_DOMAIN_ID", None)
            else:
                os.environ["ROS_DOMAIN_ID"] = prev
        res["trial"] = i
        trials.append(res)

    qs = [q for t in trials for q in t["queries"]]
    present = [q for q in qs if q["expected"] is not None]
    absent = [q for q in qs if q["expected"] is None]
    summary = {
        "planner": args.planner,
        "trials": len(trials),
        "trials_ready": sum(t["ready"] for t in trials),
        "present_pass": f"{sum(q['pass'] for q in present)}/{len(present)}",
        "absent_refused_without_motion": f"{sum(q['pass'] for q in absent)}/{len(absent)}",
        "collisions_max": max((q["collisions_total"] for q in qs), default=0),
        "surface_distance_m": sorted(q["surface_distance_m"] for q in present if "surface_distance_m" in q),
        "note": "perception stand-in + kinematic sim: chain evidence, not perception or hardware evidence",
    }
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "trials": trials}, indent=2))
    print(json.dumps(summary, indent=2))
    ok = all(q["pass"] for q in qs) and qs and all(t["ready"] for t in trials)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
