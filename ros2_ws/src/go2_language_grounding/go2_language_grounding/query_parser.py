"""Rule-based parser for natural-language navigation queries.

Handles the small grammar this project actually needs:

  "<verb> (to|near|next to|by|in front of|behind|left of|right of) the <attribute>? <noun>"
  "(the)? <attribute>? <noun> (next to|beside|in front of|...) the <ref_attribute>? <ref_noun>"

Parsed output is (target_noun, attribute, relation, reference_noun, reference_attribute, stand_off_m).

This is intentionally not an LLM — it is predictable, fast, and offline. For
queries that fall outside these forms the parser returns a best-effort noun/attribute
without a relation; the grounding node will still run CLIP scoring on the full query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedQuery:
    raw: str
    target_noun: str
    attribute: str
    relation: Optional[str]
    reference_noun: str
    reference_attribute: str
    stand_off_m: Optional[float]


# Canonical relation → synonyms / surface forms.
_RELATION_PHRASES: list[tuple[str, tuple[str, ...]]] = [
    ("in_front_of", ("in front of", "ahead of")),
    ("behind", ("behind",)),
    ("left_of", ("left of", "to the left of")),
    ("right_of", ("right of", "to the right of")),
    ("above", ("above", "on top of", "over")),
    ("below", ("below", "under", "underneath")),
    ("on", ("on",)),
    ("near", ("next to", "beside", "near", "close to", "by")),
]

# Filler verbs we can safely strip. Do NOT include phrases that contain a spatial
# relation word ("near", "next to", "beside", "toward", "by", "in front of", ...),
# or the parser will lose the relation before it can detect it.
_VERBS = (
    "go to",
    "go over",
    "go",
    "move to",
    "move",
    "approach",
    "walk to",
    "walk",
    "head to",
    "head",
    "stand",
    "go stand",
    "please",
    "can you",
    "could you",
)

# Articles / determiners.
_ARTICLES = ("the", "a", "an", "that", "this")

# Common colors used as attributes. (Extend as needed.)
_COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "beige", "tan",
}

# Descriptive adjectives we try to keep as attributes (shape/material).
_ADJECTIVES = {
    "wooden", "plastic", "metal", "metallic", "leather", "glass",
    "large", "big", "small", "tiny", "tall", "short", "long",
    "wide", "narrow", "round", "square", "rectangular", "oval",
    "empty", "full", "open", "closed", "broken", "new", "old",
    "bright", "dim",
}


def _strip_leading_verb(q: str) -> str:
    s = q
    for verb in sorted(_VERBS, key=len, reverse=True):
        if s.startswith(verb + " "):
            s = s[len(verb) + 1 :]
            break
        if s == verb:
            s = ""
            break
    return s


def _strip_article(tokens: list[str]) -> list[str]:
    if tokens and tokens[0] in _ARTICLES:
        return tokens[1:]
    return tokens


def _split_attribute_noun(tokens: list[str]) -> tuple[str, str]:
    """Given tokens after stripping article/relation, split into (attribute, noun)."""
    if not tokens:
        return "", ""
    attrs = []
    while tokens and (tokens[0] in _COLORS or tokens[0] in _ADJECTIVES):
        attrs.append(tokens.pop(0))
    # If no attribute found but token 0 is a plausible adjective (two-word), nothing fancy.
    noun = " ".join(tokens).strip()
    attribute = " ".join(attrs).strip()
    return attribute, noun


def parse(query: str) -> ParsedQuery:
    raw = query
    q = query.strip().lower()

    # Optional stand-off override at the end: "... stand off 0.6m"
    stand_off = None
    m = re.search(r"stand[- ]?off\s+(\d+(?:\.\d+)?)\s*m?$", q)
    if m:
        stand_off = float(m.group(1))
        q = q[: m.start()].strip()

    q = _strip_leading_verb(q)

    # Try binary relation: ".. <relation> the <ref>"
    for rel, phrases in _RELATION_PHRASES:
        for phrase in sorted(phrases, key=len, reverse=True):
            # Match ' <phrase> the ' or ' <phrase> a ' or ' <phrase> <token>'
            pattern = rf"\s({re.escape(phrase)})\s+"
            matches = list(re.finditer(pattern, " " + q + " "))
            if not matches:
                continue
            # Use the LAST occurrence to prefer "chair next to the table"
            # over "chair left of the right red ..."
            idx = matches[-1].start()
            # Map back to positions in q
            left_s = (" " + q + " ")[:idx].strip()
            right_s = (" " + q + " ")[idx + 1 + len(phrase) :].strip()

            left_tokens = _strip_article(left_s.split())
            right_tokens = _strip_article(right_s.split())

            if not left_tokens or not right_tokens:
                continue

            target_attr, target_noun = _split_attribute_noun(left_tokens)
            ref_attr, ref_noun = _split_attribute_noun(right_tokens)

            return ParsedQuery(
                raw=raw,
                target_noun=target_noun,
                attribute=target_attr,
                relation=rel,
                reference_noun=ref_noun,
                reference_attribute=ref_attr,
                stand_off_m=stand_off,
            )

    # Unary relation: "go <phrase> the <target>"
    for rel, phrases in _RELATION_PHRASES:
        for phrase in sorted(phrases, key=len, reverse=True):
            prefix = phrase + " "
            if q.startswith(prefix):
                rest = q[len(prefix) :].strip()
                tokens = _strip_article(rest.split())
                attr, noun = _split_attribute_noun(tokens)
                return ParsedQuery(
                    raw=raw,
                    target_noun=noun,
                    attribute=attr,
                    relation=rel,
                    reference_noun="",
                    reference_attribute="",
                    stand_off_m=stand_off,
                )

    # Fall-through: no relation. Attribute + noun from remaining tokens.
    tokens = _strip_article(q.split())
    attr, noun = _split_attribute_noun(tokens)
    return ParsedQuery(
        raw=raw,
        target_noun=noun,
        attribute=attr,
        relation=None,
        reference_noun="",
        reference_attribute="",
        stand_off_m=stand_off,
    )
