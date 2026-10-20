"""Unit tests for spatial-relation edge computation and satisfaction."""

import math

import numpy as np
import pytest
from go2_scene_graph.graph_store import TrackedObject
from go2_scene_graph.spatial_relations import (
    RelationParams,
    compute_edges,
    satisfies_relation,
)


def _obj(oid: str, x: float, y: float, z: float = 0.0, dims=(0.3, 0.3, 0.3)) -> TrackedObject:
    return TrackedObject(
        object_id=oid,
        label="thing",
        confidence=1.0,
        position_map=np.array([x, y, z], dtype=np.float32),
        dimensions=np.array(dims, dtype=np.float32),
        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        observation_count=5,
        first_seen_ns=0,
        last_seen_ns=0,
        last_depth_median_m=2.0,
    )


def test_near_emitted_when_within_threshold():
    p = RelationParams(near_threshold_m=1.0)
    a = _obj("A", 0.0, 0.0)
    b = _obj("B", 0.5, 0.0)
    edges = compute_edges([a, b], p, heading_yaw=0.0)
    near = [e for e in edges if e["relation_type"] == "near"]
    assert len(near) == 1
    assert near[0]["source_id"] == "A" and near[0]["target_id"] == "B"
    assert near[0]["distance_m"] == 0.5


def test_in_front_of_requires_forward_cone():
    p = RelationParams(lateral_cone_deg=30.0)
    a = _obj("A", 0.0, 0.0)
    # B is ahead of A, so the directed edge reads B in_front_of A.
    b = _obj("B", 1.0, 0.0)
    edges = compute_edges([a, b], p, heading_yaw=0.0)
    assert any(e["relation_type"] == "in_front_of" and e["source_id"] == "B" and e["target_id"] == "A" for e in edges)


def test_left_of_right_of_at_right_angles():
    p = RelationParams(lateral_cone_deg=30.0)
    a = _obj("A", 0.0, 0.0)
    # right_of: heading=0 (forward=+x), right=+x-rotated-CW-by-90°=-y direction.
    # Our convention: right = [sin(yaw), -cos(yaw)] → (0, -1) for yaw=0.
    # So a vector pointing in +x is NOT "right of"; a vector pointing in -y IS.
    b_right = _obj("R", 0.0, -1.0)
    b_left = _obj("L", 0.0, 1.0)
    edges = compute_edges([a, b_right, b_left], p, heading_yaw=0.0)
    assert any(e["relation_type"] == "right_of" and e["source_id"] == "R" and e["target_id"] == "A" for e in edges)
    assert any(e["relation_type"] == "left_of" and e["source_id"] == "L" and e["target_id"] == "A" for e in edges)


def test_above_and_below_use_z_axis():
    p = RelationParams(vertical_threshold_m=0.1)
    a = _obj("A", 0.0, 0.0, 0.0)
    b_high = _obj("H", 0.0, 0.0, 1.0)
    b_low = _obj("L", 0.0, 0.0, -1.0)
    edges = compute_edges([a, b_high, b_low], p, heading_yaw=0.0)
    assert any(e["relation_type"] == "above" and e["source_id"] == "H" and e["target_id"] == "A" for e in edges)
    assert any(e["relation_type"] == "below" and e["source_id"] == "L" and e["target_id"] == "A" for e in edges)


def test_satisfies_relation_near_scores_high_for_coincident():
    a = _obj("A", 0.0, 0.0)
    b = _obj("B", 0.2, 0.0)
    s = satisfies_relation(a, b, "near", RelationParams(near_threshold_m=1.0))
    assert s > 0.5


def test_satisfies_relation_near_returns_zero_for_far_away():
    a = _obj("A", 0.0, 0.0)
    b = _obj("B", 5.0, 0.0)
    s = satisfies_relation(a, b, "near", RelationParams(near_threshold_m=1.0))
    assert s == 0.0


def test_satisfies_relation_in_front_of_scales_with_alignment():
    a = _obj("A", 0.0, 0.0)
    # b exactly ahead
    b_ahead = _obj("B", 1.0, 0.0)
    # b off to the side at 45°
    b_45 = _obj("C", math.cos(math.radians(45)), math.sin(math.radians(45)))
    s_ahead = satisfies_relation(b_ahead, a, "in_front_of", RelationParams(lateral_cone_deg=50.0), heading_yaw=0.0)
    s_45 = satisfies_relation(b_45, a, "in_front_of", RelationParams(lateral_cone_deg=50.0), heading_yaw=0.0)
    assert s_ahead > s_45
    assert s_ahead > 0.0


@pytest.mark.parametrize("relation,position,heading", [
    ("in_front_of", (1., 0., 0.), 0.),
    ("behind", (-1., 0., 0.), 0.),
    ("left_of", (0., 1., 0.), 0.),
    ("right_of", (0., -1., 0.), 0.),
    ("above", (0., 0., 1.), 0.),
    ("below", (0., 0., -1.), 0.),
    ("in_front_of", (0., 1., 0.), math.pi / 2),
    ("left_of", (-1., 0., 0.), math.pi / 2),
])
def test_directed_edges_and_scores_read_subject_relation_reference(relation, position, heading):
    reference = _obj("reference", 0., 0., 0.)
    subject = _obj("subject", *position)
    edges = compute_edges([reference, subject], heading_yaw=heading)
    assert any(e["source_id"] == "subject" and e["target_id"] == "reference"
               and e["relation_type"] == relation for e in edges)
    assert not any(e["source_id"] == "reference" and e["target_id"] == "subject"
                   and e["relation_type"] == relation for e in edges)
    assert satisfies_relation(subject, reference, relation, heading_yaw=heading) > .9
    assert satisfies_relation(reference, subject, relation, heading_yaw=heading) == 0.


def test_on_requires_upper_subject_and_surface_contact():
    table = _obj("table", 0., 0., .5, dims=(1., 1., 1.))
    cup = _obj("cup", 0., 0., 1.1, dims=(.2, .2, .2))
    edges = compute_edges([table, cup])
    on = [e for e in edges if e["relation_type"] == "on"]
    assert [(e["source_id"], e["target_id"]) for e in on] == [("cup", "table")]
    assert satisfies_relation(cup, table, "on") > .9
    assert satisfies_relation(table, cup, "on") == 0.
    # A centroid inside the table is not evidence of top-surface contact.
    cup.position_map[2] = .55
    assert satisfies_relation(cup, table, "on") == 0.


@pytest.mark.parametrize("unknown", ["subject", "reference", "both"])
def test_unknown_dimensions_never_establish_on_relation(unknown):
    table = _obj("table", 0., 0., .5, dims=(1., 1., 1.))
    cup = _obj("cup", 0., 0., 1.1, dims=(.2, .2, .2))
    if unknown in ("subject", "both"):
        cup.dimensions[:] = 0
    if unknown in ("reference", "both"):
        table.dimensions[:] = 0
    assert not any(e["relation_type"] == "on" for e in compute_edges([table, cup]))
    assert satisfies_relation(cup, table, "on") == 0.
