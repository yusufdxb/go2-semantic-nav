#!/usr/bin/env python3
"""Pre-registered analysis of the NVDEC/VIC clock A/B (`jetson_lab.sh clockab`).

Implements docs/preregistration/nvdec-clock-latency.md. The
constants below are part of that registration and do not change after a run.

  clockab_analyze.py --schedule N     registered phase order for blocks 1..N, one per line
  clockab_analyze.py windows.jsonl    validity checks, primary result, decision

Exit 0 with a SUPPORTED or NULL decision, 3 when the run is INVALID (repeat
it; an invalid run is not a null), 1 when the file cannot be read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

# Registered design. Each arm twice in each position, each ordered pair of
# neighbouring phases twice: balances drift and first-order carryover.
ORDERS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")
REGISTERED_BLOCKS = 6
MEASURED_WINDOWS = 25        # first 25 windows after the 5 s settle
MIN_FPS = 12.0
AT_MAX_FRAC = 0.90           # manipulation check for a held clock
MIN_COUNTED_BLOCKS = 5
CONTRAST_MAX_FRAC = 0.50     # mean NVDEC clock in A above this fraction of max: no contrast
SESOI_MS = 2.0
BOOT_N = 10_000
BOOT_SEED = 20261002
EXACT_SIGNFLIP_MAX_N = 20    # beyond this, a seeded Monte Carlo sign flip


def schedule(n_blocks: int) -> list[str]:
    return [ORDERS[i % len(ORDERS)] for i in range(n_blocks)]


def load(path) -> tuple[dict, list[dict], list[dict]]:
    config: dict = {}
    windows: list[dict] = []
    phases: list[dict] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        if kind == "config":
            config = rec
        elif kind == "window":
            windows.append(rec)
        elif kind == "phase":
            phases.append(rec)
    return config, windows, phases


def measured(windows: list[dict], block: int, phase: str) -> list[dict]:
    rows = [w for w in windows if w["block"] == block and w["phase"] == phase and not w["settle"]]
    rows.sort(key=lambda w: w["t_s"])
    return rows[:MEASURED_WINDOWS]


def at_max_frac(rows: list[dict], engine: str) -> float:
    if not rows:
        return 0.0
    hits = [w.get(f"{engine}_hz") is not None and w.get(f"{engine}_hz") == w.get(f"{engine}_max_hz") for w in rows]
    return sum(hits) / len(hits)


def phase_problems(rows: list[dict], phase: str, config: dict) -> list[str]:
    """Why a phase fails the registered validity rules; empty when it is valid."""
    why = []
    if len(rows) < MEASURED_WINDOWS:
        why.append(f"{len(rows)} of {MEASURED_WINDOWS} measured windows")
    if any(w.get("p50_ms") is None or (w.get("fps") or 0) < MIN_FPS for w in rows):
        why.append(f"a window without frames or below {MIN_FPS:g} fps")
    if len({w.get("restarts") for w in rows}) > 1:
        why.append("camera pipeline restarted")
    if phase == "A":
        for eng in ("nvdec", "vic"):
            gov, low = config.get(f"{eng}_default_governor"), config.get(f"{eng}_default_min_hz")
            if any(w.get(f"{eng}_governor") != gov or w.get(f"{eng}_min_hz") != low for w in rows):
                why.append(f"{eng} not at its recorded default ({gov}, min {low} Hz)")
    if phase in ("B", "C") and at_max_frac(rows, "nvdec") < AT_MAX_FRAC:
        why.append(f"NVDEC at max in {at_max_frac(rows, 'nvdec'):.0%} of windows")
    if phase == "C" and at_max_frac(rows, "vic") < AT_MAX_FRAC:
        why.append(f"VIC at max in {at_max_frac(rows, 'vic'):.0%} of windows")
    return why


def bootstrap_ci(d: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(BOOT_N, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def signflip_p(d: np.ndarray) -> float:
    """Two-sided sign-flip permutation p of the mean; exact up to EXACT_SIGNFLIP_MAX_N blocks."""
    obs = abs(float(d.mean())) - 1e-12
    if len(d) <= EXACT_SIGNFLIP_MAX_N:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    else:
        signs = np.random.default_rng(BOOT_SEED).choice((1.0, -1.0), size=(100_000, len(d)))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs))


def holm(ps: dict[str, float]) -> dict[str, float]:
    out, running = {}, 0.0
    ordered = sorted(ps.items(), key=lambda kv: kv[1])
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        out[name] = running
    return out


def decide(mean: float, lo: float, hi: float) -> tuple[str, str]:
    """Registered decision. First matching rule wins."""
    if hi < 0 and mean <= -SESOI_MS:
        return "SUPPORTED", f"holding the NVDEC clock at max lowered p50 by {-mean:.2f} ms"
    if hi < 0:
        return "NULL", f"detectable but below {SESOI_MS:.1f} ms"
    if lo > 0:
        return "NULL", "slower with the clock held at max"
    if -SESOI_MS < lo and hi < SESOI_MS:
        return "NULL", "no meaningful effect"
    return "NULL", "inconclusive"


def comparison(values: dict[int, dict[str, float]], blocks: list[int], hi: str, lo: str, key: str) -> dict | None:
    d = np.array([values[b][f"{hi}_{key}"] - values[b][f"{lo}_{key}"] for b in blocks], dtype=float)
    if len(d) < 2:
        return None
    ci_lo, ci_hi = bootstrap_ci(d)
    return {"n": len(d), "mean_ms": float(d.mean()), "ci95_ms": [ci_lo, ci_hi], "p": signflip_p(d),
            "per_block_ms": [round(float(x), 3) for x in d]}


def analyse(config: dict, windows: list[dict], phases: list[dict]) -> dict:
    blocks = sorted({w["block"] for w in windows})
    deviations = []
    if len(blocks) != REGISTERED_BLOCKS:
        deviations.append(f"{len(blocks)} blocks run, {REGISTERED_BLOCKS} registered")
    if blocks and blocks != list(range(1, len(blocks) + 1)):
        deviations.append(f"block numbers {blocks}")

    table, values, problems = {}, {}, {}
    for b in blocks:
        orders = {w.get("order") for w in windows if w["block"] == b}
        want = ORDERS[(b - 1) % len(ORDERS)]
        if orders != {want}:
            deviations.append(f"block {b}: order {sorted(map(str, orders))}, registered {want}")
        values[b], problems[b], table[b] = {}, {}, {"order": "/".join(sorted(map(str, orders)))}
        for p in "ABC":
            rows = measured(windows, b, p)
            problems[b][p] = phase_problems(rows, p, config)
            p50 = [w["p50_ms"] for w in rows if w.get("p50_ms") is not None]
            p95 = [w["p95_ms"] for w in rows if w.get("p95_ms") is not None]
            nv = [w["nvdec_hz"] for w in rows if w.get("nvdec_hz") is not None]
            values[b][f"{p}_p50"] = float(np.median(p50)) if p50 else float("nan")
            values[b][f"{p}_p95"] = float(np.median(p95)) if p95 else float("nan")
            values[b][f"{p}_nvdec_mean_hz"] = float(np.mean(nv)) if nv else float("nan")
            table[b][p] = {"p50_median_ms": values[b][f"{p}_p50"], "p95_median_ms": values[b][f"{p}_p95"],
                           "windows": len(rows), "nvdec_at_max": at_max_frac(rows, "nvdec"),
                           "vic_at_max": at_max_frac(rows, "vic"), "invalid": problems[b][p]}
    power = {(r["block"], r["phase"]): r for r in phases}

    counted = [b for b in blocks if not problems[b]["A"] and not problems[b]["B"]]
    counted_c = [b for b in counted if not problems[b]["C"]]
    result = {"blocks": table, "counted_blocks": counted, "counted_blocks_c": counted_c,
              "deviations": deviations, "config": {k: v for k, v in config.items() if k != "kind"}}

    a_rows = [w for b in counted for w in measured(windows, b, "A")]
    max_hz = next((w.get("nvdec_max_hz") for w in a_rows if w.get("nvdec_max_hz")), None)
    a_clock = [w["nvdec_hz"] for w in a_rows if w.get("nvdec_hz") is not None]
    contrast = float(np.mean(a_clock)) / max_hz if a_clock and max_hz else None
    result["a_nvdec_mean_frac_of_max"] = contrast

    invalid = []
    if len(counted) < MIN_COUNTED_BLOCKS:
        invalid.append(f"{len(counted)} blocks count for B - A, need {MIN_COUNTED_BLOCKS}")
    if contrast is None or contrast > CONTRAST_MAX_FRAC:
        shown = "unknown" if contrast is None else f"{contrast:.0%}"
        invalid.append(f"no clock contrast: NVDEC in A averaged {shown} of max "
                       f"(limit {CONTRAST_MAX_FRAC:.0%}); the default governor already runs it fast")
    if invalid:
        result.update(decision="INVALID", detail="; ".join(invalid))
        return result

    primary = comparison(values, counted, "B", "A", "p50")
    lo, hi = primary["ci95_ms"]
    result["primary_B_minus_A_p50"] = primary
    result["decision"], result["detail"] = decide(primary["mean_ms"], lo, hi)

    secondary = {"B_minus_A_p95": comparison(values, counted, "B", "A", "p95"),
                 "C_minus_B_p50": comparison(values, counted_c, "C", "B", "p50"),
                 "C_minus_B_p95": comparison(values, counted_c, "C", "B", "p95")}
    secondary = {k: v for k, v in secondary.items() if v is not None}
    for name, adj in holm({k: v["p"] for k, v in secondary.items()}).items():
        secondary[name]["p_holm"] = adj
    for arm in ("B", "C"):
        pairs = [(power.get((b, arm), {}).get("vdd_in_mw_avg"), power.get((b, "A"), {}).get("vdd_in_mw_avg"))
                 for b in (counted if arm == "B" else counted_c)]
        diffs = [x - y for x, y in pairs if x is not None and y is not None]
        if diffs:
            secondary[f"power_{arm}_minus_A_mw"] = {"n": len(diffs), "mean_mw": float(np.mean(diffs))}
    tj = [r["tj_c_max"] for r in phases if r.get("tj_c_max") is not None]
    if tj:
        secondary["tj_c_range"] = [min(tj), max(tj)]
    result["secondary_exploratory"] = secondary
    return result


def report(r: dict) -> str:
    out = ["NVDEC/VIC clock A/B (docs/preregistration/nvdec-clock-latency.md)", ""]
    out.append("block order   A p50   B p50   C p50   B-A    NVDEC@max A/B/C   invalid")
    for b, t in r["blocks"].items():
        d = t["B"]["p50_median_ms"] - t["A"]["p50_median_ms"]
        bad = "; ".join(f"{p}: {', '.join(t[p]['invalid'])}" for p in "ABC" if t[p]["invalid"]) or "-"
        at_max = "/".join(f"{t[p]['nvdec_at_max']:.0%}" for p in "ABC")
        out.append(f"{b:5d} {t['order']:5s} {t['A']['p50_median_ms']:7.2f} {t['B']['p50_median_ms']:7.2f} "
                   f"{t['C']['p50_median_ms']:7.2f} {d:6.2f}   {at_max:15s}  {bad}")
    frac = r.get("a_nvdec_mean_frac_of_max")
    out += ["", f"blocks counted for B - A: {r['counted_blocks']}",
            f"NVDEC clock in A: {'unknown' if frac is None else f'{frac:.0%}'} of max on average"]
    if "primary_B_minus_A_p50" in r:
        p = r["primary_B_minus_A_p50"]
        out.append(f"PRIMARY  B - A, window p50: {p['mean_ms']:+.2f} ms, 95 % CI [{p['ci95_ms'][0]:+.2f}, "
                   f"{p['ci95_ms'][1]:+.2f}], sign-flip p = {p['p']:.3f}, n = {p['n']}")
    out.append(f"DECISION {r['decision']}: {r['detail']}")
    sec = r.get("secondary_exploratory") or {}
    if sec:
        out += ["", "EXPLORATORY (not a claim):"]
        for name, s in sec.items():
            if "ci95_ms" in s:
                out.append(f"  {name}: {s['mean_ms']:+.2f} ms [{s['ci95_ms'][0]:+.2f}, {s['ci95_ms'][1]:+.2f}], "
                           f"p = {s['p']:.3f}, Holm p = {s.get('p_holm', s['p']):.3f}, n = {s['n']}")
            elif "mean_mw" in s:
                out.append(f"  {name}: {s['mean_mw']:+.0f} mW, n = {s['n']}")
            else:
                out.append(f"  {name}: {s}")
    if r["deviations"]:
        out += ["", "DEVIATIONS from the registration:"] + [f"  {d}" for d in r["deviations"]]
    if r["config"].get("rehearsal"):
        out += ["", "REHEARSAL run (desktop, simulated clocks): not evidence about the robot."]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("windows", nargs="?", help="windows.jsonl written by jetson_lab.sh clockab")
    ap.add_argument("--schedule", type=int, metavar="N", help="print the registered order for blocks 1..N")
    ap.add_argument("--json", action="store_true", help="only the result as JSON")
    a = ap.parse_args()
    if a.schedule is not None:
        print("\n".join(schedule(a.schedule)))
        return 0
    if not a.windows:
        ap.error("give windows.jsonl or --schedule N")
    try:
        result = analyse(*load(a.windows))
    except (OSError, ValueError, KeyError) as e:
        print(f"cannot read {a.windows}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result) if a.json else report(result) + "\n\n" + json.dumps(result))
    return 3 if result["decision"] == "INVALID" else 0


if __name__ == "__main__":
    sys.exit(main())
