"""ROS test: a dry-run 'nearest chair' goal picks the closer of two equal chairs.

The real GroundingNode runs with the encoder stubbed out (lexical scoring only),
so both chairs score the same. The robot sits at the map origin via a static TF.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("go2_semantic_msgs.action")
pytest.importorskip("tf2_ros")

from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3  # noqa: E402
from go2_language_grounding.grounding_node import GroundingNode  # noqa: E402
from go2_semantic_msgs.action import GroundAndNavigate  # noqa: E402
from go2_semantic_msgs.msg import SceneGraph, SemanticObject  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from tf2_ros import StaticTransformBroadcaster  # noqa: E402


def _wait(future, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.done(), "future timed out"
    return future.result()


def _chair(object_id: str, x: float, header) -> SemanticObject:
    obj = SemanticObject()
    obj.object_id = object_id
    obj.label = "chair"
    obj.confidence = 0.9
    obj.pose = PoseStamped()
    obj.pose.header = header
    obj.pose.pose.position.x = x
    obj.pose.pose.orientation.w = 1.0
    obj.dimensions_xyz = Vector3(x=0.5, y=0.5, z=0.9)
    obj.embedding = [1.0]
    obj.embedding_dim = 1
    obj.observation_count = 3
    return obj


def _graph(stamp) -> SceneGraph:
    graph = SceneGraph()
    graph.header.stamp = stamp
    graph.header.frame_id = "map"
    # The far chair comes first, so a plain best-score pick lands on it.
    graph.nodes = [_chair("chair-far", 4.0, graph.header), _chair("chair-near", 1.5, graph.header)]
    graph.total_objects = 2
    return graph


def _costmap(stamp) -> OccupancyGrid:
    grid = OccupancyGrid()
    grid.header.stamp = stamp
    grid.header.frame_id = "map"
    grid.info.resolution = 0.1
    grid.info.width = 100
    grid.info.height = 100
    grid.info.origin.position.x = -5.0
    grid.info.origin.position.y = -5.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * (grid.info.width * grid.info.height)
    return grid


@contextmanager
def _running(monkeypatch):
    monkeypatch.setattr(GroundingNode, "_load_encoder", lambda _self: None)
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    io = Node("nearest_test_io")
    grounding = GroundingNode(parameter_overrides=[
        Parameter("max_scene_graph_age_s", value=0.5),
        Parameter("max_costmap_age_s", value=0.5),
    ])
    tf = TransformStamped()
    tf.header.stamp = io.get_clock().now().to_msg()
    tf.header.frame_id = "map"
    tf.child_frame_id = "base_link"
    tf.transform.rotation.w = 1.0
    broadcaster = StaticTransformBroadcaster(io)
    broadcaster.sendTransform(tf)
    graph_pub = io.create_publisher(SceneGraph, "/semantic/scene_graph", 10)
    cm_pub = io.create_publisher(OccupancyGrid, "/global_costmap/costmap", 10)
    client = ActionClient(io, GroundAndNavigate, "/semantic/ground_and_navigate")
    feeding = threading.Event()
    feeding.set()

    def feed():
        while feeding.is_set():
            stamp = io.get_clock().now().to_msg()
            graph_pub.publish(_graph(stamp))
            cm_pub.publish(_costmap(stamp))
            time.sleep(0.05)

    for n in (io, grounding):
        executor.add_node(n)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (
            grounding._runtime_gate_error() is not None or grounding._robot_xyz() is None
        ):
            time.sleep(0.02)
        assert grounding._runtime_gate_error() is None
        assert grounding._robot_xyz() is not None
        yield client
    finally:
        feeding.clear()
        executor.shutdown(timeout_sec=2.0)
        for n in (grounding, io):
            n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _dry_run(client, text: str):
    assert client.wait_for_server(timeout_sec=2.0)
    goal = GroundAndNavigate.Goal()
    goal.text_query = text
    goal.stand_off_m = 0.9
    goal.timeout_seconds = 5.0
    goal.dry_run = True
    goal.top_k = 5
    handle = _wait(client.send_goal_async(goal))
    assert handle.accepted
    return _wait(handle.get_result_async()).result


def test_nearest_chair_picks_the_closer_one(monkeypatch):
    with _running(monkeypatch) as client:
        plain = _dry_run(client, "go to the chair")
        nearest = _dry_run(client, "take me to the nearest chair")
    assert plain.success and nearest.success
    assert plain.chosen_object_id == "chair-far"
    assert nearest.chosen_object_id == "chair-near"
    assert "nearest match" in nearest.message
