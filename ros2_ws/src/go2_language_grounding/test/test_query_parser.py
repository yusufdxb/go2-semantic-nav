"""Unit tests for the rule-based query parser."""

import pytest
from go2_language_grounding.query_parser import parse


def test_simple_noun():
    q = parse("chair")
    assert q.target_noun == "chair"
    assert q.attribute == ""
    assert q.relation is None


def test_noun_with_article_and_attribute():
    q = parse("the red chair")
    assert q.target_noun == "chair"
    assert q.attribute == "red"
    assert q.relation is None


def test_go_near_unary_relation():
    q = parse("go near the window")
    assert q.target_noun == "window"
    assert q.relation == "near"


def test_go_stand_next_to():
    q = parse("go stand next to the red chair")
    assert q.target_noun == "chair"
    assert q.attribute == "red"
    assert q.relation == "near"


def test_binary_relation_next_to():
    q = parse("the table next to the couch")
    assert q.target_noun == "table"
    assert q.relation == "near"
    assert q.reference_noun == "couch"


def test_binary_relation_in_front_of_with_attributes():
    q = parse("the blue cup in front of the wooden table")
    assert q.target_noun == "cup"
    assert q.attribute == "blue"
    assert q.relation == "in_front_of"
    assert q.reference_noun == "table"
    assert q.reference_attribute == "wooden"


def test_left_of_and_right_of():
    q1 = parse("the chair to the left of the sofa")
    q2 = parse("the chair to the right of the sofa")
    assert q1.relation == "left_of"
    assert q2.relation == "right_of"


def test_above_below_on():
    q1 = parse("the lamp above the table")
    q2 = parse("the rug under the coffee table")
    q3 = parse("the cup on the desk")
    assert q1.relation == "above"
    assert q2.relation == "below"
    assert q3.relation == "on"


def test_stand_off_suffix():
    q = parse("go near the window stand off 0.6m")
    assert q.stand_off_m == pytest.approx(0.6)
    assert q.target_noun == "window"


def test_fallback_noise_returns_best_effort():
    q = parse("hmmm the thing over there")
    assert q.raw == "hmmm the thing over there"
    # Should at least extract some noun without crashing
    assert isinstance(q.target_noun, str)


def test_multi_word_attribute_adjective():
    # "wooden dining table" — no attribute dictionary hit on "dining", so noun is "dining table"
    q = parse("the wooden dining table")
    assert q.attribute == "wooden"
    assert "table" in q.target_noun
