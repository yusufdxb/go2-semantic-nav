"""Unit tests for spatial-relation edge computation and satisfaction."""

import math

import numpy as np

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
    # b is straight ahead of a (along +x if yaw=0)
    b = _obj("B", 1.0, 0.0)
    edges = compute_edges([a, b], p, heading_yaw=0.0)
    assert any(e["relation_type"] == "in_front_of" and e["source_id"] == "A" and e["target_id"] == "B" for e in edges)


def test_left_of_right_of_at_right_angles():
    p = RelationParams(lateral_cone_deg=30.0)
    a = _obj("A", 0.0, 0.0)
    # right_of: heading=0 (forward=+x), right=+x-rotated-CW-by-90°=-y direction.
    # Our convention: right = [sin(yaw), -cos(yaw)] → (0, -1) for yaw=0.
    # So a vector pointing in +x is NOT "right of"; a vector pointing in -y IS.
    b_right = _obj("R", 0.0, -1.0)
    b_left = _obj("L", 0.0, 1.0)
    edges = compute_edges([a, b_right, b_left], p, heading_yaw=0.0)
    assert any(e["relation_type"] == "right_of" and e["target_id"] == "R" for e in edges)
    assert any(e["relation_type"] == "left_of" and e["target_id"] == "L" for e in edges)


def test_above_and_below_use_z_axis():
    p = RelationParams(vertical_threshold_m=0.1)
    a = _obj("A", 0.0, 0.0, 0.0)
    b_high = _obj("H", 0.0, 0.0, 1.0)
    b_low = _obj("L", 0.0, 0.0, -1.0)
    edges = compute_edges([a, b_high, b_low], p, heading_yaw=0.0)
    assert any(e["relation_type"] == "above" and e["target_id"] == "H" for e in edges)
    assert any(e["relation_type"] == "below" and e["target_id"] == "L" for e in edges)


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
    s_ahead = satisfies_relation(a, b_ahead, "in_front_of", RelationParams(lateral_cone_deg=50.0), heading_yaw=0.0)
    s_45 = satisfies_relation(a, b_45, "in_front_of", RelationParams(lateral_cone_deg=50.0), heading_yaw=0.0)
    assert s_ahead > s_45
    assert s_ahead > 0.0
