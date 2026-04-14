"""Unit tests for the three-layer grounding rejection gates.

These tests exercise the pure decision logic (`accept/reject` given scored
candidates) without spinning up rclpy or running a real encoder. They codify
the v1→v2 ablation finding as an executable regression test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    """Minimal stand-in for GroundingCandidate just for the gate math."""
    score: float
    score_clip: float
    score_label: float


def _gate(
    candidates: list[Candidate],
    absolute_floor: float = 0.15,
    label_floor: float = 0.40,
    clip_floor: float = 0.22,
    margin_min: float = 0.0,
) -> tuple[bool, str]:
    """Port of GroundingNode._execute_action's rejection logic, pure-Python.

    Returns (accepted, reason).
    """
    if not candidates:
        return False, "no candidates"
    top = candidates[0]
    if top.score < absolute_floor:
        return False, f"total {top.score:.3f} below absolute_floor {absolute_floor:.2f}"
    if top.score_label < label_floor and top.score_clip < clip_floor:
        return False, f"weak on both label ({top.score_label:.2f}) and clip ({top.score_clip:.2f})"
    if margin_min > 0.0 and len(candidates) >= 2:
        margin = top.score - candidates[1].score
        if margin < margin_min:
            return False, f"margin {margin:.3f} below {margin_min:.3f}"
    return True, "accepted"


# --- v1 regression: single-threshold behavior we explicitly outgrew -----

def test_v1_behavior_single_threshold_accepts_noise():
    """Document the v1 failure mode: single threshold accepts 'person' for 'guitar'."""
    # v1 gate: just `total >= 0.15`
    guitar_vs_person = Candidate(score=0.179, score_clip=0.18, score_label=0.03)
    assert guitar_vs_person.score >= 0.15  # v1 accepts
    # But with v2 gates (label_floor=0.40, clip_floor=0.22), both floors fail
    ok, reason = _gate([guitar_vs_person])
    assert not ok
    assert "weak on both" in reason


# --- v2 accept / reject cases ------------------------------------------

def test_v2_accepts_strong_clip_match():
    c = Candidate(score=0.45, score_clip=0.40, score_label=0.10)
    ok, _ = _gate([c])
    assert ok


def test_v2_accepts_strong_label_match():
    c = Candidate(score=0.40, score_clip=0.15, score_label=0.80)
    ok, _ = _gate([c])
    assert ok


def test_v2_rejects_when_both_weak():
    c = Candidate(score=0.20, score_clip=0.20, score_label=0.10)
    ok, reason = _gate([c])
    assert not ok
    assert "weak on both" in reason


def test_v2_rejects_below_absolute_floor():
    c = Candidate(score=0.10, score_clip=0.05, score_label=0.05)
    ok, reason = _gate([c])
    assert not ok
    assert "absolute_floor" in reason


# --- margin-based rejection (opt-in) -----------------------------------

def test_margin_accepts_when_top1_beats_top2_by_margin():
    c1 = Candidate(score=0.70, score_clip=0.60, score_label=0.80)
    c2 = Candidate(score=0.40, score_clip=0.30, score_label=0.40)
    ok, _ = _gate([c1, c2], margin_min=0.10)
    assert ok


def test_margin_rejects_when_top1_and_top2_are_close():
    c1 = Candidate(score=0.45, score_clip=0.40, score_label=0.70)
    c2 = Candidate(score=0.44, score_clip=0.39, score_label=0.68)
    ok, reason = _gate([c1, c2], margin_min=0.05)
    assert not ok
    assert "margin" in reason


def test_margin_disabled_by_default():
    """margin_min=0 should accept even identical candidates."""
    c1 = Candidate(score=0.50, score_clip=0.50, score_label=0.50)
    c2 = Candidate(score=0.50, score_clip=0.50, score_label=0.50)
    ok, _ = _gate([c1, c2], margin_min=0.0)
    assert ok


# --- tuning sensitivity ------------------------------------------------

def test_label_floor_increase_tightens_rejection():
    """If the user bumps label_floor to 0.9, 'partial label match' no longer saves a weak clip."""
    c = Candidate(score=0.35, score_clip=0.10, score_label=0.70)
    ok_default, _ = _gate([c])
    assert ok_default  # 0.70 > 0.40 → label saves it
    ok_strict, reason = _gate([c], label_floor=0.90)
    assert not ok_strict
    assert "weak on both" in reason


def test_clip_floor_tightening_catches_the_dog_case():
    """The remaining honesty failure ('the dog' → person with clip~0.23) — bumping clip_floor catches it."""
    dog_vs_person = Candidate(score=0.22, score_clip=0.23, score_label=0.03)
    ok_default, _ = _gate([dog_vs_person])
    assert ok_default  # clip 0.23 > default clip_floor 0.22
    ok_strict, reason = _gate([dog_vs_person], clip_floor=0.30)
    assert not ok_strict
    assert "weak on both" in reason
