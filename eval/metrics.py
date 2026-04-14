"""Evaluation metrics for language-grounding and navigation success."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class QueryOutcome:
    query_id: str
    query: str
    scene: str
    expected_label: Optional[str]
    success: bool
    chosen_label: Optional[str]
    top5_labels: list[str]
    grounding_score: float
    grounding_latency_ms: float
    reached: bool = False
    path_length_m: float = 0.0
    shortest_path_m: float = 0.0
    final_distance_m: float = float("inf")


def grounding_top1(outcomes: Iterable[QueryOutcome]) -> float:
    xs = [o for o in outcomes if o.expected_label is not None]
    if not xs:
        return 0.0
    return sum(1 for o in xs if o.chosen_label == o.expected_label) / len(xs)


def grounding_top5(outcomes: Iterable[QueryOutcome]) -> float:
    xs = [o for o in outcomes if o.expected_label is not None]
    if not xs:
        return 0.0
    return sum(1 for o in xs if o.expected_label in o.top5_labels) / len(xs)


def expected_failure_honesty(outcomes: Iterable[QueryOutcome]) -> float:
    """Among queries whose expected_label is None (i.e., out-of-prompt-set),
    measures how often the system correctly reports success=false rather than
    picking a wrong goal."""
    xs = [o for o in outcomes if o.expected_label is None]
    if not xs:
        return 0.0
    return sum(1 for o in xs if not o.success) / len(xs)


def navigation_success_rate(outcomes: Iterable[QueryOutcome], stand_off_m: float = 0.9, slack_m: float = 0.3) -> float:
    xs = [o for o in outcomes if o.expected_label is not None]
    if not xs:
        return 0.0
    return sum(1 for o in xs if o.reached and o.final_distance_m <= stand_off_m + slack_m) / len(xs)


def spl(outcomes: Iterable[QueryOutcome]) -> float:
    """Success weighted by Path Length (Anderson et al. 2018)."""
    xs = [o for o in outcomes if o.expected_label is not None]
    if not xs:
        return 0.0
    total = 0.0
    for o in xs:
        if not o.reached or o.path_length_m <= 0 or o.shortest_path_m <= 0:
            continue
        total += min(1.0, o.shortest_path_m / max(o.shortest_path_m, o.path_length_m))
    return total / len(xs)


def latency_percentile(outcomes: Iterable[QueryOutcome], p: float) -> float:
    xs = sorted(o.grounding_latency_ms for o in outcomes)
    if not xs:
        return 0.0
    idx = int(p * (len(xs) - 1) / 100.0)
    return xs[idx]


def summarize(outcomes: list[QueryOutcome]) -> dict:
    return {
        "n_total": len(outcomes),
        "grounding_top1": grounding_top1(outcomes),
        "grounding_top5": grounding_top5(outcomes),
        "expected_failure_honesty": expected_failure_honesty(outcomes),
        "navigation_success_rate": navigation_success_rate(outcomes),
        "spl": spl(outcomes),
        "latency_ms_p50": latency_percentile(outcomes, 50),
        "latency_ms_p95": latency_percentile(outcomes, 95),
    }
