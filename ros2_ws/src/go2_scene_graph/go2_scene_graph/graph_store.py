"""In-memory store for the open-vocabulary 3D semantic scene graph.

Design goals:
- Pure-python, no rclpy dependency — testable in isolation.
- O(N) association per detection, where N is the number of tracked objects.
  (OK for <~200 objects. If this becomes a hot spot, swap in a spatial index.)
- EMA-based pose / embedding updates to smooth noise.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class TrackedObject:
    object_id: str
    label: str
    confidence: float
    position_map: np.ndarray             # shape (3,) in meters, map frame
    dimensions: np.ndarray                # shape (3,) in meters
    embedding: np.ndarray                 # shape (D,), L2-normalized
    observation_count: int
    first_seen_ns: int
    last_seen_ns: int
    last_depth_median_m: float
    label_history: dict[str, float] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.shape[0])


@dataclass
class AssociationParams:
    assoc_max_dist_m: float = 0.5
    assoc_embed_threshold: float = 0.80
    pose_ema_alpha: float = 0.4            # weight on new observation
    dims_ema_alpha: float = 0.2
    embed_ema_alpha: float = 0.2
    object_ttl_s: float = 45.0
    min_observations_to_publish: int = 3
    merge_on_label_match_bonus: float = 0.05


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


class SceneGraphStore:
    """Holds tracked objects and applies data-association logic."""

    def __init__(self, params: AssociationParams | None = None) -> None:
        self._params = params or AssociationParams()
        self._objects: dict[str, TrackedObject] = {}

    # --- accessors -------------------------------------------------------

    @property
    def params(self) -> AssociationParams:
        return self._params

    def update_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self._params, k):
                setattr(self._params, k, v)

    def all_objects(self) -> list[TrackedObject]:
        return list(self._objects.values())

    def objects_with_min_observations(self, min_obs: Optional[int] = None) -> list[TrackedObject]:
        m = min_obs if min_obs is not None else self._params.min_observations_to_publish
        return [o for o in self._objects.values() if o.observation_count >= m]

    def get(self, object_id: str) -> Optional[TrackedObject]:
        return self._objects.get(object_id)

    def __len__(self) -> int:
        return len(self._objects)

    # --- associate / merge / create -------------------------------------

    def ingest(
        self,
        label: str,
        score: float,
        position_map: np.ndarray,
        dimensions: np.ndarray,
        embedding: np.ndarray,
        depth_median_m: float,
        stamp_ns: int,
    ) -> tuple[str, bool]:
        """Associate a detection with an existing object or create a new one.

        Returns (object_id, is_new).
        """
        p = self._params
        if embedding.size == 0:
            raise ValueError("ingest: embedding must be non-empty")
        # L2-normalize defensively
        norm = np.linalg.norm(embedding) + 1e-9
        embedding = (embedding / norm).astype(np.float32)

        match_id = self._find_best_match(label, position_map, embedding)
        if match_id is None:
            obj = TrackedObject(
                object_id=uuid.uuid4().hex,
                label=label,
                confidence=float(score),
                position_map=position_map.astype(np.float32).copy(),
                dimensions=dimensions.astype(np.float32).copy(),
                embedding=embedding.copy(),
                observation_count=1,
                first_seen_ns=stamp_ns,
                last_seen_ns=stamp_ns,
                last_depth_median_m=float(depth_median_m),
                label_history={label: float(score)},
            )
            self._objects[obj.object_id] = obj
            return obj.object_id, True

        self._merge(match_id, label, score, position_map, dimensions, embedding, depth_median_m, stamp_ns)
        return match_id, False

    def _find_best_match(
        self, label: str, position_map: np.ndarray, embedding: np.ndarray
    ) -> Optional[str]:
        p = self._params
        best_id: Optional[str] = None
        best_score = -math.inf
        for obj_id, obj in self._objects.items():
            dist = float(np.linalg.norm(obj.position_map - position_map))
            if dist > p.assoc_max_dist_m:
                continue
            sim = cosine_similarity(obj.embedding, embedding)
            bonus = p.merge_on_label_match_bonus if obj.label == label else 0.0
            gated = sim + bonus
            if gated < p.assoc_embed_threshold:
                continue
            score = gated - 0.1 * (dist / max(p.assoc_max_dist_m, 1e-6))
            if score > best_score:
                best_score = score
                best_id = obj_id
        return best_id

    def _merge(
        self,
        object_id: str,
        label: str,
        score: float,
        position_map: np.ndarray,
        dimensions: np.ndarray,
        embedding: np.ndarray,
        depth_median_m: float,
        stamp_ns: int,
    ) -> None:
        p = self._params
        obj = self._objects[object_id]

        a_pos = p.pose_ema_alpha
        a_dim = p.dims_ema_alpha
        a_emb = p.embed_ema_alpha

        obj.position_map = ((1 - a_pos) * obj.position_map + a_pos * position_map).astype(np.float32)
        # dimensions may be noisy; cap the update
        obj.dimensions = ((1 - a_dim) * obj.dimensions + a_dim * dimensions).astype(np.float32)

        new_embed = (1 - a_emb) * obj.embedding + a_emb * embedding
        norm = np.linalg.norm(new_embed) + 1e-9
        obj.embedding = (new_embed / norm).astype(np.float32)

        obj.observation_count += 1
        obj.last_seen_ns = stamp_ns
        obj.last_depth_median_m = float(depth_median_m)

        # Running mean confidence
        obj.confidence = float((1 - a_pos) * obj.confidence + a_pos * score)

        obj.label_history[label] = obj.label_history.get(label, 0.0) + float(score)
        # Promote the dominant label as the object's canonical label when a new label
        # beats the current one cumulatively.
        dominant = max(obj.label_history.items(), key=lambda kv: kv[1])
        obj.label = dominant[0]

    # --- maintenance ----------------------------------------------------

    def prune_stale(self, now_ns: int) -> list[str]:
        """Remove objects not seen within object_ttl_s. Returns removed ids."""
        ttl_ns = int(self._params.object_ttl_s * 1e9)
        to_remove = [oid for oid, o in self._objects.items() if now_ns - o.last_seen_ns > ttl_ns]
        for oid in to_remove:
            del self._objects[oid]
        return to_remove

    def clear(self) -> None:
        self._objects.clear()

    # --- query ----------------------------------------------------------

    def rank_by_text(self, text_embedding: np.ndarray, top_k: int = 5) -> list[tuple[TrackedObject, float]]:
        """Return top-k objects ranked by cosine to text embedding."""
        pairs: list[tuple[TrackedObject, float]] = []
        for obj in self._objects.values():
            if obj.observation_count < self._params.min_observations_to_publish:
                continue
            sim = cosine_similarity(obj.embedding, text_embedding)
            pairs.append((obj, sim))
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs[: max(0, top_k)]

    def filter_by_label(self, substring: str) -> list[TrackedObject]:
        s = substring.strip().lower()
        if not s:
            return list(self._objects.values())
        return [o for o in self._objects.values() if s in o.label.lower()]
