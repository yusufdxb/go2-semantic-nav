"""Unit tests for scene graph JSON snapshots."""

import json
import os

import numpy as np
from go2_scene_graph.graph_store import AssociationParams, SceneGraphStore
from go2_scene_graph.snapshot_io import latest_snapshot, update_seen, write_json_atomic

S = 1_000_000_000


def _store_with(labels_positions, stamp_ns=1 * S, ttl_s=45.0):
    store = SceneGraphStore(AssociationParams(min_observations_to_publish=1, object_ttl_s=ttl_s))
    for i, (label, pos) in enumerate(labels_positions):
        emb = np.zeros(8, dtype=np.float32)
        emb[i] = 1.0
        store.ingest(label=label, score=0.8, position_map=np.array(pos, dtype=np.float32),
                     dimensions=np.array([0.5, 0.5, 1.0], dtype=np.float32), embedding=emb,
                     depth_median_m=2.0, stamp_ns=stamp_ns)
    return store


def test_latest_snapshot_is_json_and_drops_embeddings():
    store = _store_with([("chair", [1.0, 0.0, 0.4])])
    snap = latest_snapshot(store.all_objects(), [], "map", 5 * S)
    obj = snap["objects"][0]
    assert snap["total_objects"] == 1 and snap["map_frame"] == "map"
    assert obj["label"] == "chair" and obj["position_map"] == [1.0, 0.0, 0.4]
    assert "embedding" not in obj and obj["embedding_dim"] == 8
    json.dumps(snap)  # serializable: no numpy types left


def test_seen_keeps_pruned_objects_and_marks_them_not_live():
    store = _store_with([("chair", [1.0, 0.0, 0.4]), ("table", [3.0, 1.0, 0.4])], ttl_s=10.0)
    seen = update_seen({}, latest_snapshot(store.all_objects(), [], "map", 1 * S))
    assert {r["label"] for r in seen.values()} == {"chair", "table"}

    store.ingest(label="chair", score=0.8, position_map=np.array([1.0, 0.0, 0.4], dtype=np.float32),
                 dimensions=np.array([0.5, 0.5, 1.0], dtype=np.float32),
                 embedding=np.eye(8, dtype=np.float32)[0], depth_median_m=2.0, stamp_ns=20 * S)
    store.prune_stale(now_ns=20 * S)  # table last seen at 1 s, ttl 10 s -> pruned
    update_seen(seen, latest_snapshot(store.all_objects(), [], "map", 20 * S))
    live = {r["label"]: r["live"] for r in seen.values()}
    assert live == {"chair": True, "table": False}


def test_write_json_atomic_replaces_and_leaves_no_temp(tmp_path):
    path = os.path.join(tmp_path, "sub", "g.json")
    write_json_atomic(path, {"a": 1})
    write_json_atomic(path, {"a": 2})
    assert json.load(open(path)) == {"a": 2}
    assert os.listdir(os.path.dirname(path)) == ["g.json"]
