#!/usr/bin/env python3
"""Registered analysis for docs/preregistration/2026-09-24-real-frames-2d-grounding.md.

Inputs: eval/results/real_frames_2d/queries.csv (model output) and
eval/real_frames/box_verdicts.csv (blind judgement of each chosen box).
Primary: BA(gate) - BA(sham), 95% block bootstrap (60 s blocks), plus the
viewpoint-cluster bootstrap added as a pre-inference deviation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "eval/results/real_frames_2d"
N_BOOT = 10_000
N_SHAM = 2000


def load():
    verdict = {(r["frame"], r["top_box"], r["query_class"]): int(r["box_contains_class"])
               for r in csv.DictReader(open(REPO / "eval/real_frames/box_verdicts.csv"))}
    rows = []
    for r in csv.DictReader(open(RES / "queries.csv")):
        has_top = r["top_box"] != ""
        present = r["truth"] == "present"
        top_ok = verdict[(r["frame"], r["top_box"], r["query_class"])] if (present and has_top) else 0
        rows.append(dict(bag=r["bag"], block=r["block"], viewpoint=r["viewpoint"], cls=r["query_class"],
                         in_vocab=r["in_vocab"] == "True", present=present, has_top=has_top,
                         top_ok=top_ok, gate_refuse=r["gate_refusal"] != ""))
    return rows


def arm_ba(rows, rng=None):
    """Balanced accuracy for every arm on a list of query rows."""
    P = [r for r in rows if r["present"]]
    A = [r for r in rows if not r["present"]]
    out = {}
    # ungated: accept top-1 whenever there is one
    out["ungated"] = (np.mean([r["top_ok"] for r in P]), np.mean([not r["has_top"] for r in A]))
    out["gate"] = (np.mean([r["top_ok"] and not r["gate_refuse"] for r in P]), np.mean([r["gate_refuse"] for r in A]))
    out["oracle"] = (np.mean([r["top_ok"] for r in P]), 1.0)
    # sham: refuse with the gate's realised refusal rate within the same bag, content-blind.
    rate = {}
    for bag in {r["bag"] for r in rows}:
        b = [r for r in rows if r["bag"] == bag]
        rate[bag] = np.mean([r["gate_refuse"] for r in b])
    # exact expectation over the sham's randomness
    out["sham"] = (np.mean([(1 - rate[r["bag"]]) * r["top_ok"] for r in P]),
                   np.mean([max(rate[r["bag"]], float(not r["has_top"])) for r in A]))
    return {k: dict(present_success=float(p), absent_refusal=float(a), ba=float((p + a) / 2)) for k, (p, a) in out.items()}, rate


def sham_draws(rows, rate, rng):
    """Monte-Carlo sham, used to print its realised per-bag refusal rate (A7 check)."""
    realised = defaultdict(list)
    bas = []
    for _ in range(N_SHAM):
        ps, ab = [], []
        per_bag = defaultdict(list)
        for r in rows:
            refuse = rng.random() < rate[r["bag"]] or not r["has_top"]
            per_bag[r["bag"]].append(refuse)
            (ps if r["present"] else ab).append((r["top_ok"] and not refuse) if r["present"] else refuse)
        bas.append((np.mean(ps) + np.mean(ab)) / 2)
        for b, v in per_bag.items():
            realised[b].append(np.mean(v))
    return float(np.mean(bas)), {b: float(np.mean(v)) for b, v in realised.items()}


def bootstrap(rows, key, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    ids = sorted(groups)
    diffs, gaps = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        sample = [r for i in pick for r in groups[ids[i]]]
        if not any(r["present"] for r in sample) or all(r["present"] for r in sample):
            continue
        m, _ = arm_ba(sample)
        diffs.append(m["gate"]["ba"] - m["sham"]["ba"])
        gaps.append(m["oracle"]["ba"] - m["gate"]["ba"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"unit": key, "n_units": len(ids), "n_boot_valid": len(diffs),
            "gate_minus_sham_ci95": q(diffs), "oracle_minus_gate_ci95": q(gaps)}


def main() -> int:
    rng = np.random.default_rng(20260924)
    rows = load()
    metrics, rate = arm_ba(rows)
    mc_ba, realised = sham_draws(rows, rate, rng)
    res = {
        "n_queries": len(rows),
        "n_present": sum(r["present"] for r in rows),
        "n_absent": sum(not r["present"] for r in rows),
        "arms": metrics,
        "primary_gate_minus_sham": metrics["gate"]["ba"] - metrics["sham"]["ba"],
        "a7_dose_check": {b: {"gate_refusal_rate": float(rate[b]), "sham_realised_rate": realised[b]} for b in rate},
        "sham_ba_monte_carlo": mc_ba,
        "ci_block_60s": bootstrap(rows, "block", rng),
        "ci_viewpoint": bootstrap(rows, "viewpoint", rng),
    }
    # exploratory, labelled as such
    per_class = {}
    for c in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == c]
        P = [r for r in rs if r["present"]]
        A = [r for r in rs if not r["present"]]
        per_class[c] = {
            "n_present": len(P), "n_absent": len(A),
            "top1_box_correct": float(np.mean([r["top_ok"] for r in P])) if P else None,
            "gate_present_success": float(np.mean([r["top_ok"] and not r["gate_refuse"] for r in P])) if P else None,
            "gate_accept_rate_present": float(np.mean([not r["gate_refuse"] for r in P])) if P else None,
            "gate_absent_refusal": float(np.mean([r["gate_refuse"] for r in A])) if A else None,
        }
    A = [r for r in rows if not r["present"]]
    res["exploratory"] = {
        "per_class": per_class,
        "absent_refusal_in_vocab": float(np.mean([r["gate_refuse"] for r in A if r["in_vocab"]])),
        "absent_refusal_out_of_vocab": float(np.mean([r["gate_refuse"] for r in A if not r["in_vocab"]])),
    }
    lo_b = res["ci_block_60s"]["gate_minus_sham_ci95"][0]
    lo_v = res["ci_viewpoint"]["gate_minus_sham_ci95"][0]
    d = res["primary_gate_minus_sham"]
    res["verdict"] = ("DEAD: CI includes 0" if min(lo_b, lo_v) <= 0 else
                      "SUPPORTED at registered magnitude (>=0.15)" if d >= 0.15 else
                      "EFFECT REAL but below registered 0.15")
    (RES / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "exploratory"}, indent=2))
    print(json.dumps(res["exploratory"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
