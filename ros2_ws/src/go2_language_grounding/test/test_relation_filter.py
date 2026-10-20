"""Grounding consumes directed source-relation-target graph edges."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("go2_semantic_msgs.msg")
pytest.importorskip("go2_scene_graph.spatial_relations")

from go2_language_grounding.grounding_node import GroundingNode
from go2_language_grounding.query_parser import parse
from go2_language_grounding.scoring import ScoreWeights
from go2_scene_graph.graph_store import TrackedObject
from go2_scene_graph.spatial_relations import compute_edges
from go2_semantic_msgs.msg import GroundingCandidate, SceneGraph, SceneGraphEdge, SemanticObject


def tracked(object_id, label, xyz):
    return TrackedObject(object_id, label, .9, np.asarray(xyz, dtype=np.float32),
                         np.ones(3, dtype=np.float32), np.array([1., 0.]), 3, 0, 0, 2.)


def graph_for(objects):
    graph = SceneGraph()
    graph.nodes = [SemanticObject(object_id=o.object_id, label=o.label) for o in objects]
    graph.edges = [SceneGraphEdge(source_object_id=e["source_id"], target_object_id=e["target_id"],
                                 relation_type=e["relation_type"], distance_m=e["distance_m"],
                                 relation_score=e["relation_score"])
                   for e in compute_edges(objects)]
    return graph


@pytest.mark.parametrize("relation,position", [
    ("left of", (0., 1., 0.)),
    ("right of", (0., -1., 0.)),
    ("in front of", (1., 0., 0.)),
    ("behind", (-1., 0., 0.)),
    ("above", (0., 0., 1.)),
    ("below", (0., 0., -1.)),
    ("on", (0., 0., 1.)),
])
def test_relation_selects_physically_correct_subject(relation, position):
    reference = tracked("table", "table", (0., 0., 0.))
    correct = tracked("correct", "chair", position)
    wrong = tracked("wrong", "chair", tuple(-v for v in position))
    graph = graph_for([reference, wrong, correct])
    candidates = [GroundingCandidate(object_id=o.object_id, label=o.label, score_label=1.)
                  for o in (wrong, correct)]
    target = SimpleNamespace(_encode_text_cached=lambda text: None,
                             _current_weights=lambda: ScoreWeights())
    filtered = GroundingNode._apply_relation_filter(target, candidates, parse("chair " + relation + " table"), graph)
    assert [c.object_id for c in filtered] == ["correct"]
    assert filtered[0].score_spatial == 1.


@pytest.mark.parametrize("reverse", [False, True])
def test_near_edge_is_symmetric_regardless_of_object_order(reverse):
    reference = tracked("table", "table", (0., 0., 0.))
    subject = tracked("chair", "chair", (.4, 0., 0.))
    graph = graph_for([subject, reference] if reverse else [reference, subject])
    target = SimpleNamespace(_encode_text_cached=lambda text: None,
                             _current_weights=lambda: ScoreWeights())
    candidate = GroundingCandidate(object_id="chair", label="chair", score_label=1.)
    filtered = GroundingNode._apply_relation_filter(target, [candidate], parse("chair near table"), graph)
    assert [c.object_id for c in filtered] == ["chair"]
    assert filtered[0].score_spatial == 1.
