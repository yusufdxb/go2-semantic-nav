"""ROS integration tests for the semantic action to Nav2 action handoff.

Each test runs the real GroundingNode against a scripted fake NavigateToPose
server in one MultiThreadedExecutor, so success, abort, cancel, stale-input and
interlock paths are exercised through real action plumbing.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Optional

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("go2_semantic_msgs.action")
pytest.importorskip("nav2_msgs.action")

from geometry_msgs.msg import PoseStamped, Vector3  # noqa: E402
from go2_language_grounding.grounding_node import GroundingNode  # noqa: E402
from go2_semantic_msgs.action import GroundAndNavigate  # noqa: E402
from go2_semantic_msgs.msg import SceneGraph, SemanticObject  # noqa: E402
from nav2_msgs.action import NavigateToPose  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from rclpy.action import ActionClient, ActionServer, CancelResponse  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402


def _wait(future, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.done(), "future timed out"
    return future.result()


def _wait_for(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "condition not reached"


def _scene_graph(stamp) -> SceneGraph:
    graph = SceneGraph()
    graph.header.stamp = stamp
    graph.header.frame_id = "map"
    obj = SemanticObject()
    obj.object_id = "person-1"
    obj.label = "person"
    obj.confidence = 0.9
    obj.pose = PoseStamped()
    obj.pose.header = graph.header
    obj.pose.pose.position.x = 1.0
    obj.pose.pose.orientation.w = 1.0
    obj.dimensions_xyz = Vector3(x=0.5, y=0.5, z=1.7)
    obj.embedding = [1.0]
    obj.embedding_dim = 1
    obj.observation_count = 3
    graph.nodes = [obj]
    graph.total_objects = 1
    return graph


def _free_costmap(stamp) -> OccupancyGrid:
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


class FakeNav2:
    """Scripted NavigateToPose server.

    mode: "succeed" | "abort" | "hold" (run until canceled).
    accept_cancel: whether cancel requests are acknowledged.
    """

    def __init__(self, node: Node, mode: str, accept_cancel: bool = True) -> None:
        self.mode = mode
        self.accept_cancel = accept_cancel
        self.goals_received = 0
        self.cancel_requests = 0
        self.stopped = threading.Event()
        self.server = ActionServer(
            node,
            NavigateToPose,
            "/navigate_to_pose",
            execute_callback=self._execute,
            cancel_callback=self._on_cancel,
        )

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        self.cancel_requests += 1
        return CancelResponse.ACCEPT if self.accept_cancel else CancelResponse.REJECT

    def _execute(self, goal_handle):
        self.goals_received += 1
        if self.mode == "succeed":
            goal_handle.succeed()
        elif self.mode == "abort":
            goal_handle.abort()
        else:
            while not self.stopped.is_set():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return NavigateToPose.Result()
                time.sleep(0.02)
            goal_handle.abort()
        return NavigateToPose.Result()


class Harness:
    def __init__(self, io_node: Node, grounding: GroundingNode, nav: Optional[FakeNav2]):
        self.io_node = io_node
        self.grounding = grounding
        self.nav = nav
        self.graph_pub = io_node.create_publisher(SceneGraph, "/semantic/scene_graph", 10)
        self.costmap_pub = io_node.create_publisher(
            OccupancyGrid, "/global_costmap/costmap", 10
        )
        self.client = ActionClient(io_node, GroundAndNavigate, "/semantic/ground_and_navigate")
        self.feeding = threading.Event()
        self._feeder = threading.Thread(target=self._feed, daemon=True)

    def _feed(self) -> None:
        while self.feeding.is_set():
            stamp = self.io_node.get_clock().now().to_msg()
            self.graph_pub.publish(_scene_graph(stamp))
            self.costmap_pub.publish(_free_costmap(stamp))
            time.sleep(0.05)

    def start_inputs(self) -> None:
        self.feeding.set()
        self._feeder.start()
        _wait_for(lambda: self.grounding._runtime_gate_error() is None)

    def stop_inputs(self) -> None:
        self.feeding.clear()

    def send(self, dry_run: bool = False, timeout_s: float = 5.0):
        assert self.client.wait_for_server(timeout_sec=2.0)
        request = GroundAndNavigate.Goal()
        request.text_query = "person"
        request.stand_off_m = 0.9
        request.timeout_seconds = timeout_s
        request.dry_run = dry_run
        request.top_k = 5
        return _wait(self.client.send_goal_async(request))


@contextmanager
def _harness(monkeypatch, nav_mode: Optional[str], accept_cancel: bool = True, **params):
    monkeypatch.setattr(GroundingNode, "_load_encoder", lambda _self: None)
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=6)
    nav_node = Node("fake_nav2_server")
    io_node = Node("semantic_navigation_test_client")
    nav = FakeNav2(nav_node, nav_mode, accept_cancel) if nav_mode else None
    overrides = {
        "allow_goal_publication": True,
        "navigation_backend": "nav2_action",
        "navigation_server_timeout_s": 1.0,
        "navigation_cancel_timeout_s": 1.0,
        "max_scene_graph_age_s": 0.5,
        "max_costmap_age_s": 0.5,
    }
    overrides.update(params)
    grounding = GroundingNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in overrides.items()]
    )
    harness = Harness(io_node, grounding, nav)
    for node in (nav_node, io_node, grounding):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        harness.start_inputs()
        yield harness
    finally:
        harness.stop_inputs()
        if nav is not None:
            nav.stopped.set()
            nav.server.destroy()
        executor.shutdown(timeout_sec=2.0)
        for node in (grounding, io_node, nav_node):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_success_requires_nav2_success(monkeypatch):
    with _harness(monkeypatch, "succeed") as h:
        goal_handle = h.send()
        assert goal_handle.accepted
        result = _wait(goal_handle.get_result_async()).result
        assert result.success
        assert result.terminal_state == "REACHED"
        assert result.chosen_object_label == "person"
        assert h.nav.goals_received == 1


def test_nav2_abort_is_reported_as_failure(monkeypatch):
    with _harness(monkeypatch, "abort") as h:
        result = _wait(h.send().get_result_async()).result
        assert not result.success
        assert result.terminal_state == "NAV_ABORTED"


def test_missing_nav2_server_fails_closed(monkeypatch):
    with _harness(monkeypatch, None) as h:
        result = _wait(h.send().get_result_async()).result
        assert not result.success
        assert result.terminal_state == "NAV_UNAVAILABLE"


def test_interlock_off_rejects_live_goal_and_allows_dry_run(monkeypatch):
    with _harness(monkeypatch, "succeed", allow_goal_publication=False) as h:
        assert not h.send(dry_run=False).accepted
        dry = h.send(dry_run=True)
        assert dry.accepted
        result = _wait(dry.get_result_async()).result
        assert result.success
        assert result.terminal_state == "DRY_RUN_COMPLETE"
        assert h.nav.goals_received == 0


def test_client_cancel_propagates_to_nav2(monkeypatch):
    with _harness(monkeypatch, "hold") as h:
        goal_handle = h.send()
        _wait_for(lambda: h.nav.goals_received == 1)
        _wait(goal_handle.cancel_goal_async())
        result = _wait(goal_handle.get_result_async()).result
        assert result.terminal_state == "CANCELED"
        assert "acknowledged" in result.message
        assert h.nav.cancel_requests == 1


def test_stale_inputs_cancel_navigation(monkeypatch):
    with _harness(monkeypatch, "hold") as h:
        goal_handle = h.send()
        _wait_for(lambda: h.nav.goals_received == 1)
        h.stop_inputs()
        result = _wait(goal_handle.get_result_async()).result
        assert result.terminal_state == "INPUTS_UNAVAILABLE"
        assert "stale" in result.message
        assert h.nav.cancel_requests == 1


def test_unacknowledged_cancel_is_reported(monkeypatch):
    with _harness(monkeypatch, "hold", accept_cancel=False) as h:
        goal_handle = h.send()
        _wait_for(lambda: h.nav.goals_received == 1)
        _wait(goal_handle.cancel_goal_async())
        result = _wait(goal_handle.get_result_async()).result
        assert result.terminal_state == "CANCELED"
        assert "NOT acknowledge" in result.message
