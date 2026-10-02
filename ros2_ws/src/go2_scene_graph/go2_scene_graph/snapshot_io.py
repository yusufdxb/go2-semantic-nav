"""Write the scene graph to disk as JSON (pure python, no rclpy).

Two files in ``snapshot_dir``:

- ``scene_graph_latest.json``: the objects published right now, with edges.
- ``scene_graph_seen.json``: every object ever published, keyed by id, with
  its last known state and ``live`` false once the store pruned it. The store
  drops an object ``object_ttl_s`` after it was last seen, so this is the only
  record of objects that left the camera's view.

Embeddings are left out (512 floats per object); ``embedding_dim`` is kept.
Writes go to a temp file and are renamed into place, so a reader or a crash
never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Iterable

from .graph_store import TrackedObject


def object_record(obj: TrackedObject) -> dict[str, Any]:
    ranked = sorted(obj.label_history.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "object_id": obj.object_id,
        "label": obj.label,
        "confidence": round(float(obj.confidence), 4),
        "position_map": [round(float(v), 4) for v in obj.position_map],
        "dimensions": [round(float(v), 4) for v in obj.dimensions],
        "observation_count": int(obj.observation_count),
        "first_seen_ns": int(obj.first_seen_ns),
        "last_seen_ns": int(obj.last_seen_ns),
        "last_depth_median_m": round(float(obj.last_depth_median_m), 4),
        "label_history": {k: round(float(v), 4) for k, v in ranked},
        "embedding_dim": int(obj.embedding_dim),
    }


def latest_snapshot(
    objects: Iterable[TrackedObject], edges: list[dict], map_frame: str, stamp_ns: int
) -> dict[str, Any]:
    nodes = [object_record(o) for o in objects]
    return {
        "map_frame": map_frame,
        "stamp_ns": int(stamp_ns),
        "total_objects": len(nodes),
        "objects": nodes,
        "edges": [
            {
                "source_id": e["source_id"],
                "target_id": e["target_id"],
                "relation_type": e["relation_type"],
                "distance_m": round(float(e["distance_m"]), 4),
                "relation_score": round(float(e["relation_score"]), 4),
            }
            for e in edges
        ],
    }


def update_seen(seen: dict[str, dict], latest: dict[str, Any]) -> dict[str, dict]:
    """Merge the latest snapshot into the archive in place; returns it."""
    live_ids = {o["object_id"] for o in latest["objects"]}
    for rec in seen.values():
        rec["live"] = rec["object_id"] in live_ids
    for o in latest["objects"]:
        seen[o["object_id"]] = {**o, "live": True}
    return seen


def write_json_atomic(path: str, data: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
