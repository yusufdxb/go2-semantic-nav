"""Unit tests for scoring helpers."""

import numpy as np
import pytest
from go2_language_grounding.scoring import (
    ScoreWeights,
    combine,
    cosine_similarity,
    lexical_label_match,
)


def test_lexical_exact_match():
    assert lexical_label_match("chair", "chair") == 1.0


def test_lexical_substring_match():
    assert lexical_label_match("chair", "armchair") == pytest.approx(0.7)


def test_lexical_no_overlap():
    # Even unrelated words share some character overlap in this char-set fallback;
    # the score must be strictly less than the substring-match 0.7.
    score = lexical_label_match("chair", "window")
    assert score < 0.7


def test_cosine_identity():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_combine_default_weights():
    # Default weights are 0.7 clip + 0.2 label + 0.1 spatial
    s = combine(1.0, 1.0, 1.0)
    assert s == pytest.approx(1.0)
    s = combine(0.0, 0.0, 0.0)
    assert s == pytest.approx(0.0)


def test_combine_custom_weights():
    w = ScoreWeights(clip=1.0, label=0.0, spatial=0.0)
    assert combine(0.5, 1.0, 1.0, w) == pytest.approx(0.5)
