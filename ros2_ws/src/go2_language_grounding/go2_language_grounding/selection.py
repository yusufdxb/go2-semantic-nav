"""Choosing which scored candidate to go to. Pure: no ROS, no encoder.

By default the best-scoring candidate wins. For "the nearest chair" the robot
should instead go to the closest object that is still a confident match, so:

  1. a candidate is eligible only if it passes the same floors as the top one
     (absolute floor, and label-OR-CLIP floor), and
  2. its score is within `score_window` of the top score, so a table that just
     clears the CLIP floor cannot win "nearest chair" by being closer.

Among eligible candidates the one with the smallest planar distance to the robot
wins. Without a robot position there is no "nearest", and the top candidate is
returned with `used_distance=False` so the caller can say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class GateParams:
    absolute_floor: float = 0.15
    label_floor: float = 0.40
    clip_floor: float = 0.30


def passes_gates(score: float, score_label: float, score_clip: float, gates: GateParams) -> bool:
    if score < gates.absolute_floor:
        return False
    if score_label < gates.label_floor and score_clip < gates.clip_floor:
        return False
    return True


def planar_distance(a_xy: Sequence[float], b_xy: Sequence[float]) -> float:
    return math.hypot(float(a_xy[0]) - float(b_xy[0]), float(a_xy[1]) - float(b_xy[1]))


def choose_nearest(
    candidates: Sequence,
    robot_xy: Optional[Sequence[float]],
    gates: GateParams,
    score_window: float,
    position_of,
) -> tuple[int, bool]:
    """Index of the candidate to use, and whether distance decided it.

    `candidates` must be sorted by score, best first, and the first one must
    already have passed the gates. `position_of(candidate)` returns its (x, y).
    """
    if not candidates:
        raise ValueError("choose_nearest needs at least one candidate")
    if robot_xy is None:
        return 0, False
    top_score = float(candidates[0].score)
    best_i, best_d = 0, planar_distance(position_of(candidates[0]), robot_xy)
    for i, c in enumerate(candidates[1:], start=1):
        if top_score - float(c.score) > score_window:
            continue
        if not passes_gates(float(c.score), float(c.score_label), float(c.score_clip), gates):
            continue
        d = planar_distance(position_of(c), robot_xy)
        if d < best_d:
            best_i, best_d = i, d
    return best_i, True
