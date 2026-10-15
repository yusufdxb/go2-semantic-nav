"""Local ROS callback checks for stale objects when detections stop arriving."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("go2_semantic_msgs.msg")

from go2_scene_graph.graph_store import AssociationParams, SceneGraphStore
from go2_scene_graph.scene_graph_node import SceneGraphNode
from go2_scene_graph.spatial_relations import RelationParams
from go2_semantic_msgs.srv import QueryObjects


@pytest.fixture
def callbacks(tmp_path):
    # Exercise the actual callbacks without creating DDS participants or
    # depending on a running clock, TF publisher, or detector.
    from builtin_interfaces.msg import Time

    store = SceneGraphStore(AssociationParams(object_ttl_s=1., min_observations_to_publish=1))
    store.ingest("chair", .9, np.array([1., 0., .5]), np.ones(3), np.array([1., 0.]), 2., 0)
    published = []
    clock = SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=2_000_000_000,
                                                       to_msg=lambda: Time(sec=2)))
    params = {"map_frame": "map", "min_observations_to_publish": 1}
    target = SimpleNamespace(_store=store, _relation_params=RelationParams(),
                             get_clock=lambda: clock,
                             get_parameter=lambda key: SimpleNamespace(value=params[key]),
                             _pub_graph=SimpleNamespace(publish=published.append),
                             _pub_markers=SimpleNamespace(publish=lambda msg: None),
                             _snapshot_dir=str(tmp_path), _seen={})
    return target, published


def test_publish_expires_objects_without_new_detections(callbacks):
    target, published = callbacks
    SceneGraphNode._publish_snapshot(target)
    assert len(target._store) == 0
    assert published[0].total_objects == 0


def test_disk_snapshot_expires_objects_without_new_detections(callbacks):
    target, _ = callbacks
    SceneGraphNode._write_snapshot(target)
    with open(target._snapshot_dir + "/scene_graph_latest.json") as file:
        assert json.load(file)["objects"] == []


def test_query_expires_objects_without_new_detections(callbacks):
    target, _ = callbacks
    response = SceneGraphNode._on_query_objects(
        target, QueryObjects.Request(text_query="chair", top_k=5), QueryObjects.Response())
    assert response.success
    assert response.candidates == []
