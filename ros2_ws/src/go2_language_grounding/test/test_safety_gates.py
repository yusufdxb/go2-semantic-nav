"""Tests for fail-closed runtime input gates."""

from go2_language_grounding.safety_gates import (
    frame_error,
    freshness_error,
    publication_error,
)


def test_missing_input_fails_closed():
    assert freshness_error(now_ns=10, received_ns=0, max_age_s=3.0, label="costmap")


def test_stale_input_fails_closed():
    error = freshness_error(
        now_ns=5_000_000_001,
        received_ns=1_000_000_000,
        max_age_s=4.0,
        label="scene graph",
    )
    assert error is not None and "stale" in error


def test_recent_input_passes():
    assert freshness_error(
        now_ns=3_000_000_000,
        received_ns=1_000_000_000,
        max_age_s=3.0,
        label="costmap",
    ) is None


def test_future_receipt_time_fails_closed():
    assert freshness_error(
        now_ns=1,
        received_ns=2,
        max_age_s=3.0,
        label="costmap",
    )


def test_frame_must_be_present_and_match():
    assert frame_error(actual="", expected="map", label="costmap")
    assert frame_error(actual="odom", expected="map", label="costmap")
    assert frame_error(actual="map", expected="map", label="costmap") is None


def test_goal_publication_requires_explicit_interlock():
    assert publication_error(dry_run=False, allow_goal_publication=False)
    assert publication_error(dry_run=True, allow_goal_publication=False) is None
    assert publication_error(dry_run=False, allow_goal_publication=True) is None
