"""Clock A/B tooling: the registered analysis on synthetic runs with a known answer, and the window records."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NV_MIN, NV_MAX = 115_200_000, 858_000_000
VIC_MIN, VIC_MAX = 115_200_000, 704_000_000


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ana = _load("clockab_analyze")
lw = _load("latency_window")


def synth_run(effect_b=0.0, effect_c=0.0, noise=0.3, seed=0, blocks=6, b_held=True, a_clock=NV_MIN,
              restart_block=None, rehearsal=False):
    """A run in the registered design; block-to-block drift is shared by a block's three phases."""
    rng = np.random.default_rng(seed)
    lines = [{"kind": "config", "method": "governor", "blocks": blocks, "rehearsal": rehearsal,
              "nvdec_default_governor": "tegra_wmark", "nvdec_default_min_hz": NV_MIN, "nvdec_max_hz": NV_MAX,
              "vic_default_governor": "tegra_wmark", "vic_default_min_hz": VIC_MIN, "vic_max_hz": VIC_MAX}]
    for b, order in enumerate(ana.schedule(blocks), start=1):
        drift = rng.normal(0.0, 1.0)
        for p in order:
            nv_held, vic_held = p in "BC" and b_held, p == "C"
            base = 16.0 + drift + {"A": 0.0, "B": effect_b, "C": effect_b + effect_c}[p]
            for k in range(30):  # 5 settle windows, then 25 measured
                t = k + 0.5
                lines.append({
                    "kind": "window", "block": b, "phase": p, "order": order, "t_s": t, "settle": t < 5,
                    "fps": 15, "frames": 1000 + 15 * k,
                    "restarts": 1 if (b == restart_block and p == "B" and k > 20) else 0,
                    "p50_ms": base + rng.normal(0, noise), "p95_ms": base + 4 + rng.normal(0, noise),
                    "max_ms": base + 8,
                    "nvdec_governor": "performance" if p in "BC" else "tegra_wmark",
                    "nvdec_hz": NV_MAX if nv_held else a_clock, "nvdec_min_hz": NV_MIN, "nvdec_max_hz": NV_MAX,
                    "vic_governor": "performance" if vic_held else "tegra_wmark",
                    "vic_hz": VIC_MAX if vic_held else VIC_MIN, "vic_min_hz": VIC_MIN, "vic_max_hz": VIC_MAX})
            lines.append({"kind": "phase", "block": b, "phase": p, "order": order, "method": "governor",
                          "tegrastats_samples": 30, "tj_c_max": 50.0,
                          "vdd_in_mw_avg": 6000 + (400 if p in "BC" else 0) + (200 if p == "C" else 0)})
    return lines


def run(tmp_path, lines):
    path = tmp_path / "windows.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path, ana.analyse(*ana.load(path))


def test_schedule_balances_position_and_carryover():
    orders = ana.schedule(6)
    for pos in range(3):
        assert sorted(o[pos] for o in orders) == ["A", "A", "B", "B", "C", "C"]
    pairs = [o[i:i + 2] for o in orders for i in range(2)]
    assert sorted(set(pairs)) == ["AB", "AC", "BA", "BC", "CA", "CB"]
    assert all(pairs.count(x) == 2 for x in set(pairs))
    assert ana.schedule(7)[6] == ana.ORDERS[0]


def test_large_effect_is_supported(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=-4.0))
    p = r["primary_B_minus_A_p50"]
    assert r["decision"] == "SUPPORTED"
    assert abs(p["mean_ms"] + 4.0) < 0.3 and p["ci95_ms"][1] < 0
    assert p["n"] == 6 and p["p"] == 2 / 64  # every block negative: the exact floor at n = 6


def test_no_effect_is_a_null(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=0.0))
    assert (r["decision"], r["detail"]) == ("NULL", "no meaningful effect")


def test_small_real_effect_is_a_null_below_threshold(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=-1.0))
    assert (r["decision"], r["detail"]) == ("NULL", "detectable but below 2.0 ms")


def test_slower_is_its_own_null(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=+1.0))
    assert (r["decision"], r["detail"]) == ("NULL", "slower with the clock held at max")


def test_clock_write_that_did_not_take_is_invalid_not_null(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=0.0, b_held=False))
    assert r["decision"] == "INVALID" and "0 blocks count" in r["detail"]
    assert "primary_B_minus_A_p50" not in r
    assert any("NVDEC at max in 0%" in x for x in r["blocks"][1]["B"]["invalid"])


def test_no_clock_contrast_is_invalid(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=-4.0, a_clock=NV_MAX))
    assert r["decision"] == "INVALID" and "no clock contrast" in r["detail"]


def test_restarted_phase_drops_its_block_only(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=-4.0, restart_block=3))
    assert r["counted_blocks"] == [1, 2, 4, 5, 6]
    assert "camera pipeline restarted" in r["blocks"][3]["B"]["invalid"]
    assert r["decision"] == "SUPPORTED" and r["primary_B_minus_A_p50"]["n"] == 5


def test_too_few_blocks_is_invalid_and_reported(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=-4.0, blocks=4))
    assert r["decision"] == "INVALID"
    assert "4 blocks run, 6 registered" in r["deviations"]


def test_order_deviation_is_reported(tmp_path):
    lines = synth_run(effect_b=-4.0)
    for x in lines:
        if x.get("block") == 2:
            x["order"] = "ABC"
    _, r = run(tmp_path, lines)
    assert any(d.startswith("block 2: order") for d in r["deviations"])


def test_secondary_is_labelled_and_holm_adjusted(tmp_path):
    _, r = run(tmp_path, synth_run(effect_b=-4.0, effect_c=-1.0))
    sec = r["secondary_exploratory"]
    assert set(sec) >= {"B_minus_A_p95", "C_minus_B_p50", "C_minus_B_p95", "power_B_minus_A_mw"}
    assert all(sec[k]["p_holm"] >= sec[k]["p"] for k in ("B_minus_A_p95", "C_minus_B_p50", "C_minus_B_p95"))
    assert sec["power_B_minus_A_mw"]["mean_mw"] == 400
    assert "EXPLORATORY (not a claim)" in ana.report(r)


def test_signflip_and_holm():
    assert ana.signflip_p(np.full(6, -1.0)) == 2 / 64
    assert ana.signflip_p(np.array([1.0, -1.0] * 3)) == 1.0
    assert ana.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_cli_exit_codes(tmp_path):
    script = str(HERE / "clockab_analyze.py")
    sched = subprocess.run([sys.executable, script, "--schedule", "6"], capture_output=True, text=True)
    assert sched.stdout.split() == list(ana.ORDERS)
    good, _ = run(tmp_path, synth_run(effect_b=-4.0, rehearsal=True))
    res = subprocess.run([sys.executable, script, str(good)], capture_output=True, text=True)
    assert res.returncode == 0 and "DECISION SUPPORTED" in res.stdout and "REHEARSAL run" in res.stdout
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(x) for x in synth_run(b_held=False)) + "\n")
    assert subprocess.run([sys.executable, script, str(bad)], capture_output=True).returncode == 3


def test_window_record_marks_settle_and_missing_latency():
    clocks = {"governor": "performance", "cur_freq": NV_MAX, "min_freq": NV_MIN, "max_freq": NV_MAX}
    stats = {"fps": 15, "frames": 10, "restarts": 0, "arrival_to_publish_ms": {"p50": 16.5, "p95": 21.0, "max": 30.1}}
    early = lw.window_record(stats, t_s=2.0, settle_s=5.0, block=1, phase="B", order="ABC", nvdec=clocks, vic={})
    late = lw.window_record({"fps": 0, "frames": 10, "restarts": 0}, t_s=6.0, settle_s=5.0, block=1, phase="B",
                            order="ABC", nvdec=clocks, vic={})
    assert early["settle"] and early["p50_ms"] == 16.5 and early["nvdec_hz"] == NV_MAX and early["vic_hz"] is None
    assert not late["settle"] and late["p50_ms"] is None
    s = lw.summarize([early, late])
    assert s["windows"] == 1 and s["settle_windows"] == 1 and s["p50_median_ms"] is None and s["nvdec_at_max"] == 1.0
    assert lw.summarize([])["windows"] == 0


def test_read_devfreq(tmp_path):
    d = tmp_path / "15480000.nvdec"
    d.mkdir()
    for name, value in (("governor", "tegra_wmark"), ("cur_freq", NV_MIN), ("min_freq", NV_MIN), ("max_freq", NV_MAX)):
        (d / name).write_text(f"{value}\n")
    assert lw.read_devfreq(str(d)) == {"governor": "tegra_wmark", "cur_freq": NV_MIN, "min_freq": NV_MIN,
                                       "max_freq": NV_MAX}
    assert lw.read_devfreq(str(tmp_path / "missing"))["cur_freq"] is None
    assert lw.read_devfreq("") == {}


def test_lab_scripts_parse():
    for name in ("jetson_lab.sh", "run_lab.sh"):
        assert subprocess.run(["bash", "-n", str(HERE / name)]).returncode == 0, name
