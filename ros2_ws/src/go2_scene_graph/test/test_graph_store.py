"""Unit tests for SceneGraphStore."""

import numpy as np

from go2_scene_graph.graph_store import (
    AssociationParams,
    SceneGraphStore,
    cosine_similarity,
)


def _unit_vec(*components) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def test_create_single_object():
    store = SceneGraphStore()
    oid, is_new = store.ingest(
        label="chair",
        score=0.9,
        position_map=np.array([1.0, 0.0, 0.5], dtype=np.float32),
        dimensions=np.array([0.4, 0.4, 1.0], dtype=np.float32),
        embedding=_unit_vec(1.0, 0.2, 0.1),
        depth_median_m=2.1,
        stamp_ns=1_000_000_000,
    )
    assert is_new
    assert len(store) == 1
    obj = store.get(oid)
    assert obj is not None and obj.label == "chair" and obj.observation_count == 1


def test_merge_same_object_same_embedding_close_position():
    store = SceneGraphStore()
    e = _unit_vec(0.9, 0.1, 0.05)
    oid1, _ = store.ingest("chair", 0.9, np.array([1.0, 0.0, 0.5]), np.array([0.4, 0.4, 1.0]), e, 2.1, 1)
    oid2, is_new = store.ingest("chair", 0.85, np.array([1.05, 0.02, 0.5]), np.array([0.4, 0.4, 1.0]), e, 2.1, 2)
    assert not is_new and oid1 == oid2
    assert len(store) == 1
    obj = store.get(oid1)
    assert obj.observation_count == 2


def test_new_object_when_embedding_differs():
    store = SceneGraphStore(AssociationParams(assoc_embed_threshold=0.8))
    e1 = _unit_vec(1.0, 0.0, 0.0)
    e2 = _unit_vec(0.0, 1.0, 0.0)
    oid1, _ = store.ingest("chair", 0.9, np.array([1.0, 0.0, 0.5]), np.array([0.4, 0.4, 1.0]), e1, 2.1, 1)
    oid2, is_new = store.ingest("chair", 0.9, np.array([1.05, 0.0, 0.5]), np.array([0.4, 0.4, 1.0]), e2, 2.1, 2)
    assert is_new
    assert oid1 != oid2
    assert len(store) == 2


def test_new_object_when_position_far_even_if_embedding_matches():
    store = SceneGraphStore(AssociationParams(assoc_max_dist_m=0.5))
    e = _unit_vec(0.9, 0.1, 0.05)
    oid1, _ = store.ingest("chair", 0.9, np.array([0.0, 0.0, 0.5]), np.array([0.4, 0.4, 1.0]), e, 2.1, 1)
    oid2, is_new = store.ingest("chair", 0.9, np.array([3.0, 0.0, 0.5]), np.array([0.4, 0.4, 1.0]), e, 2.1, 2)
    assert is_new and oid1 != oid2 and len(store) == 2


def test_pose_ema_smoothes_position():
    # assoc_max_dist_m must exceed the 1.0 m jump so the second ingest is a merge, not a new object.
    store = SceneGraphStore(AssociationParams(pose_ema_alpha=0.4, assoc_max_dist_m=2.0))
    e = _unit_vec(1.0, 0.0, 0.0)
    store.ingest("chair", 0.9, np.array([0.0, 0.0, 0.0]), np.array([0.4, 0.4, 1.0]), e, 2.0, 1)
    oid, is_new = store.ingest("chair", 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.4, 0.4, 1.0]), e, 2.0, 2)
    assert not is_new
    obj = store.get(oid)
    # EMA with alpha=0.4 → 0.4 * 1.0 + 0.6 * 0.0 = 0.4
    assert abs(obj.position_map[0] - 0.4) < 1e-5


def test_pruning_removes_stale_objects():
    store = SceneGraphStore(AssociationParams(object_ttl_s=1.0))
    e = _unit_vec(1.0, 0.0, 0.0)
    oid1, _ = store.ingest("chair", 0.9, np.array([0.0, 0.0, 0.0]), np.array([0.4, 0.4, 1.0]), e, 2.0, 0)
    removed = store.prune_stale(now_ns=int(2 * 1e9))
    assert oid1 in removed
    assert len(store) == 0


def test_rank_by_text_respects_min_observations():
    store = SceneGraphStore(AssociationParams(min_observations_to_publish=3))
    e_chair = _unit_vec(1.0, 0.0, 0.0)
    e_table = _unit_vec(0.0, 1.0, 0.0)
    # chair observed 3 times, table observed 1 time
    for i in range(3):
        store.ingest("chair", 0.9, np.array([1.0, 0.0, 0.5]), np.array([0.4, 0.4, 1.0]), e_chair, 2.1, i)
    store.ingest("table", 0.9, np.array([3.0, 0.0, 0.5]), np.array([0.4, 0.4, 1.0]), e_table, 2.1, 10)

    pairs = store.rank_by_text(e_chair, top_k=5)
    assert len(pairs) == 1
    assert pairs[0][0].label == "chair"


def test_cosine_similarity_edge_cases():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    assert abs(cosine_similarity(a, c) + 1.0) < 1e-6
    # mismatched shapes → 0
    assert cosine_similarity(a, np.array([1.0, 0.0])) == 0.0
