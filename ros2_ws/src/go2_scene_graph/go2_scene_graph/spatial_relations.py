"""Geometric spatial-relation computations on the scene graph.

All inputs are in the map frame. The "robot heading" parameter (optional) is used
for viewpoint-dependent relations like left_of / right_of / in_front_of / behind.
If heading is not provided, those relations are computed relative to the map +X axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .graph_store import TrackedObject


@dataclass
class RelationParams:
    near_threshold_m: float = 1.2
    lateral_cone_deg: float = 35.0
    vertical_threshold_m: float = 0.15
    on_vertical_slack_m: float = 0.10
    max_edges: int = 200


def _robot_yaw_frame(heading_yaw: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (forward_unit, right_unit) 2D vectors for left/right/front/back tests."""
    if heading_yaw is None:
        heading_yaw = 0.0
    fwd = np.array([math.cos(heading_yaw), math.sin(heading_yaw)], dtype=np.float32)
    right = np.array([math.sin(heading_yaw), -math.cos(heading_yaw)], dtype=np.float32)
    return fwd, right


def _horizontal_offset(a: TrackedObject, b: TrackedObject) -> np.ndarray:
    """Returns the 2D (x, y) offset from a to b."""
    return (b.position_map[:2] - a.position_map[:2]).astype(np.float32)


def euclidean_distance(a: TrackedObject, b: TrackedObject) -> float:
    return float(np.linalg.norm(a.position_map - b.position_map))


def compute_edges(
    objects: Iterable[TrackedObject],
    params: RelationParams | None = None,
    heading_yaw: Optional[float] = None,
) -> list[dict]:
    """Compute pairwise spatial-relation edges.

    Returns a list of dicts with keys {source_id, target_id, relation_type,
    distance_m, relation_score}. Each pair generates at most a handful of edges,
    capped by params.max_edges overall.
    """
    p = params or RelationParams()
    fwd, right = _robot_yaw_frame(heading_yaw)
    cone_cos = math.cos(math.radians(p.lateral_cone_deg))

    objs = list(objects)
    edges: list[dict] = []

    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i == j:
                continue
            dist = euclidean_distance(a, b)
            if dist > max(p.near_threshold_m, p.vertical_threshold_m) * 3.0:
                continue
            off_xy = _horizontal_offset(a, b)
            off_xy_norm = float(np.linalg.norm(off_xy))
            dz = float(b.position_map[2] - a.position_map[2])

            # Near (undirected — emit as a→b only; downstream can densify)
            if dist <= p.near_threshold_m and i < j:
                edges.append(
                    dict(
                        source_id=a.object_id,
                        target_id=b.object_id,
                        relation_type="near",
                        distance_m=dist,
                        relation_score=float(max(0.0, 1.0 - dist / p.near_threshold_m)),
                    )
                )

            # Viewpoint-dependent relations need non-zero horizontal offset
            if off_xy_norm > 1e-3:
                unit = off_xy / off_xy_norm
                fwd_dot = float(np.dot(unit, fwd))
                right_dot = float(np.dot(unit, right))

                if fwd_dot >= cone_cos:
                    edges.append(
                        dict(
                            source_id=a.object_id,
                            target_id=b.object_id,
                            relation_type="in_front_of",
                            distance_m=dist,
                            relation_score=float((fwd_dot - cone_cos) / (1 - cone_cos + 1e-9)),
                        )
                    )
                elif -fwd_dot >= cone_cos:
                    edges.append(
                        dict(
                            source_id=a.object_id,
                            target_id=b.object_id,
                            relation_type="behind",
                            distance_m=dist,
                            relation_score=float((-fwd_dot - cone_cos) / (1 - cone_cos + 1e-9)),
                        )
                    )

                if right_dot >= cone_cos:
                    edges.append(
                        dict(
                            source_id=a.object_id,
                            target_id=b.object_id,
                            relation_type="right_of",
                            distance_m=dist,
                            relation_score=float((right_dot - cone_cos) / (1 - cone_cos + 1e-9)),
                        )
                    )
                elif -right_dot >= cone_cos:
                    edges.append(
                        dict(
                            source_id=a.object_id,
                            target_id=b.object_id,
                            relation_type="left_of",
                            distance_m=dist,
                            relation_score=float((-right_dot - cone_cos) / (1 - cone_cos + 1e-9)),
                        )
                    )

            # Above / below / on
            if dz > p.vertical_threshold_m:
                edges.append(
                    dict(
                        source_id=a.object_id,
                        target_id=b.object_id,
                        relation_type="above",
                        distance_m=dist,
                        relation_score=float(min(1.0, dz / (p.vertical_threshold_m * 4.0))),
                    )
                )
            elif dz < -p.vertical_threshold_m:
                edges.append(
                    dict(
                        source_id=a.object_id,
                        target_id=b.object_id,
                        relation_type="below",
                        distance_m=dist,
                        relation_score=float(min(1.0, -dz / (p.vertical_threshold_m * 4.0))),
                    )
                )

            # On: a sits on b → a is slightly above b AND xy-close AND |dz| ~ b.dim.z/2
            if (
                0.0 < dz < p.on_vertical_slack_m + b.dimensions[2] / 2.0 + a.dimensions[2] / 2.0
                and off_xy_norm < (b.dimensions[0] + b.dimensions[1]) / 4.0 + 0.15
            ):
                edges.append(
                    dict(
                        source_id=a.object_id,
                        target_id=b.object_id,
                        relation_type="on",
                        distance_m=dist,
                        relation_score=float(max(0.0, 1.0 - off_xy_norm)),
                    )
                )

            if len(edges) >= p.max_edges:
                return edges

    return edges


def satisfies_relation(
    subject: TrackedObject,
    reference: TrackedObject,
    relation: str,
    params: RelationParams | None = None,
    heading_yaw: Optional[float] = None,
) -> float:
    """Return a [0, 1] goodness score for whether subject stands in `relation` to reference.

    0 means "relation clearly violated"; 1 means "relation clearly satisfied".
    """
    p = params or RelationParams()
    fwd, right = _robot_yaw_frame(heading_yaw)
    cone_cos = math.cos(math.radians(p.lateral_cone_deg))

    off_xy = (reference.position_map[:2] - subject.position_map[:2]).astype(np.float32)
    off_norm = float(np.linalg.norm(off_xy))
    dist = euclidean_distance(subject, reference)
    dz = float(reference.position_map[2] - subject.position_map[2])

    if relation in ("near", "beside"):
        return float(max(0.0, 1.0 - dist / p.near_threshold_m)) if dist <= p.near_threshold_m else 0.0

    if off_norm < 1e-3 and relation in ("left_of", "right_of", "in_front_of", "behind"):
        return 0.0
    unit = off_xy / max(off_norm, 1e-6)

    if relation == "in_front_of":
        d = float(np.dot(unit, fwd))
        return float(max(0.0, (d - cone_cos) / (1 - cone_cos + 1e-9)))
    if relation == "behind":
        d = float(np.dot(unit, fwd))
        return float(max(0.0, (-d - cone_cos) / (1 - cone_cos + 1e-9)))
    if relation == "right_of":
        d = float(np.dot(unit, right))
        return float(max(0.0, (d - cone_cos) / (1 - cone_cos + 1e-9)))
    if relation == "left_of":
        d = float(np.dot(unit, right))
        return float(max(0.0, (-d - cone_cos) / (1 - cone_cos + 1e-9)))

    if relation == "above":
        return float(max(0.0, min(1.0, dz / (p.vertical_threshold_m * 4.0))))
    if relation == "below":
        return float(max(0.0, min(1.0, -dz / (p.vertical_threshold_m * 4.0))))

    if relation == "on":
        if (
            0.0 < dz < p.on_vertical_slack_m + reference.dimensions[2] / 2.0 + subject.dimensions[2] / 2.0
            and off_norm < (reference.dimensions[0] + reference.dimensions[1]) / 4.0 + 0.15
        ):
            return float(max(0.0, 1.0 - off_norm))
        return 0.0

    # Unknown relation → no opinion
    return 0.5
