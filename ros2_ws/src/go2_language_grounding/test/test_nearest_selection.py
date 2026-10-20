"""'Take me to the nearest chair': parsing and candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

from go2_language_grounding.query_parser import parse
from go2_language_grounding.selection import GateParams, choose_nearest, passes_gates


@dataclass
class Cand:
    label: str
    score: float
    score_clip: float
    score_label: float
    x: float
    y: float


def _pos(c: Cand):
    return (c.x, c.y)


GATES = GateParams()


# --- parser ---------------------------------------------------------------

def test_nearest_chair_sets_flag_and_keeps_noun():
    q = parse("take me to the nearest chair")
    assert q.nearest is True
    assert q.target_noun == "chair"
    assert q.relation is None


def test_closest_is_a_synonym():
    q = parse("go to the closest red chair")
    assert q.nearest is True
    assert (q.attribute, q.target_noun) == ("red", "chair")


def test_nearest_is_not_the_near_relation():
    q = parse("nearest table")
    assert q.relation is None
    assert q.target_noun == "table"


def test_near_relation_still_parses_without_flag():
    q = parse("go near the table")
    assert q.nearest is False
    assert q.relation == "near"
    assert q.target_noun == "table"


def test_nearest_with_relation():
    q = parse("the nearest chair next to the table")
    assert q.nearest is True
    assert q.relation == "near"
    assert (q.target_noun, q.reference_noun) == ("chair", "table")


def test_plain_query_unchanged():
    q = parse("go stand next to the red chair")
    assert q.nearest is False
    assert q.relation == "near"


# --- selection ------------------------------------------------------------

def test_picks_closer_of_two_equally_good_chairs():
    far = Cand("chair", 0.80, 0.70, 1.0, 5.0, 0.0)
    near = Cand("chair", 0.78, 0.68, 1.0, 1.0, 0.0)
    idx, used = choose_nearest([far, near], (0.0, 0.0), GATES, 0.10, _pos)
    assert used is True
    assert idx == 1


def test_closer_non_chair_outside_window_does_not_win():
    chair = Cand("chair", 0.80, 0.70, 1.0, 5.0, 0.0)
    table = Cand("table", 0.45, 0.35, 0.0, 0.5, 0.0)   # passes CLIP floor, far below top
    idx, _ = choose_nearest([chair, table], (0.0, 0.0), GATES, 0.10, _pos)
    assert idx == 0


def test_closer_candidate_failing_gates_does_not_win():
    chair = Cand("chair", 0.30, 0.25, 0.5, 5.0, 0.0)
    weak = Cand("stool", 0.28, 0.20, 0.10, 0.5, 0.0)   # within window, weak on both
    assert not passes_gates(weak.score, weak.score_label, weak.score_clip, GATES)
    idx, _ = choose_nearest([chair, weak], (0.0, 0.0), GATES, 0.10, _pos)
    assert idx == 0


def test_no_robot_pose_falls_back_to_best_match():
    far = Cand("chair", 0.80, 0.70, 1.0, 5.0, 0.0)
    near = Cand("chair", 0.78, 0.68, 1.0, 1.0, 0.0)
    idx, used = choose_nearest([far, near], None, GATES, 0.10, _pos)
    assert (idx, used) == (0, False)


def test_single_candidate():
    only = Cand("chair", 0.8, 0.7, 1.0, 3.0, 4.0)
    assert choose_nearest([only], (0.0, 0.0), GATES, 0.10, _pos) == (0, True)
