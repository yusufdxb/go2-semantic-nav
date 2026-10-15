"""Pure runtime-safety checks for grounding inputs."""

from __future__ import annotations


def freshness_error(
    *, now_ns: int, received_ns: int, max_age_s: float, label: str
) -> str | None:
    """Return a fail-closed error for missing, future, or stale input."""
    if received_ns <= 0:
        return f"{label} not received"
    age_ns = now_ns - received_ns
    if age_ns < 0:
        return f"{label} receipt time is in the future"
    if max_age_s > 0.0 and age_ns > int(max_age_s * 1e9):
        return f"{label} stale ({age_ns / 1e9:.2f}s > {max_age_s:.2f}s)"
    return None


def frame_error(*, actual: str, expected: str, label: str) -> str | None:
    """Return a fail-closed error when an input frame is absent or unexpected."""
    if not actual:
        return f"{label} frame is empty"
    if actual != expected:
        return f"{label} frame {actual!r} does not match map frame {expected!r}"
    return None


def publication_error(*, dry_run: bool, allow_goal_publication: bool) -> str | None:
    """Require an explicit operator interlock before publishing a motion goal."""
    if not dry_run and not allow_goal_publication:
        return "goal publication is disabled; use dry_run or enable the operator interlock"
    return None
