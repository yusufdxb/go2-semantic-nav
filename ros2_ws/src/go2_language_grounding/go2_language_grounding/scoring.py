"""Grounding scoring: combine CLIP similarity, label lexical match, and spatial fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreWeights:
    clip: float = 0.7
    label: float = 0.2
    spatial: float = 0.1


def lexical_label_match(target_noun: str, object_label: str) -> float:
    """Simple substring match with partial credit for shared prefixes."""
    if not target_noun or not object_label:
        return 0.0
    t = target_noun.strip().lower()
    lbl = object_label.strip().lower()
    if t == lbl:
        return 1.0
    if t in lbl or lbl in t:
        return 0.7
    # Character overlap fallback
    t_set = set(t)
    l_set = set(lbl)
    if not t_set or not l_set:
        return 0.0
    return 0.3 * (len(t_set & l_set) / len(t_set | l_set))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def combine(
    clip_score: float,
    label_score: float,
    spatial_score: float,
    weights: ScoreWeights | None = None,
) -> float:
    w = weights or ScoreWeights()
    return float(w.clip * clip_score + w.label * label_score + w.spatial * spatial_score)
