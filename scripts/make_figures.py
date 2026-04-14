#!/usr/bin/env python3
"""Generate paper / README figures from eval artifacts.

Inputs:
    eval/results/*/summary.json   (grounding eval summaries)
    eval/results/latencies_*.json (per-stage latency JSONs)
    eval/results/thermal/thermal.csv (Jetson thermal samples)

Outputs (to --out):
    latency_per_stage.png   bar chart of p50/p95 per stage per platform
    grounding_ablation.png  bar chart of honesty + top-1 per config
    thermal_plot.png        FPS + GPU temp + GPU util vs. wall time

Depends on matplotlib only. Safe to run on machines without GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib required: pip install matplotlib")


def _load_jsons(glob_root: Path, name_glob: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(glob_root.rglob(name_glob)):
        try:
            out[str(p)] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return out


def plot_latency_per_stage(latency_files: dict[str, dict], out: Path) -> None:
    if not latency_files:
        return
    stages = ["latency_detector_ms", "latency_segmenter_ms", "latency_encoder_ms",
              "latency_backproject_ms", "latency_total_ms"]
    platforms = list(latency_files.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(platforms), 1)
    x = list(range(len(stages)))
    for i, (path, stats) in enumerate(latency_files.items()):
        label = Path(path).parent.name + "/" + Path(path).stem
        p50 = [stats.get(s, {}).get("p50", 0.0) for s in stages]
        p95 = [stats.get(s, {}).get("p95", 0.0) for s in stages]
        offset = width * (i - len(platforms) / 2 + 0.5)
        ax.bar([xi + offset for xi in x], p50, width=width, label=f"{label} p50")
        ax.errorbar([xi + offset for xi in x], p50,
                    yerr=[[0] * len(stages), [a - b for a, b in zip(p95, p50)]],
                    fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("latency_", "").replace("_ms", "") for s in stages])
    ax.set_ylabel("latency (ms)")
    ax.set_title("Per-stage latency (p50 bar, p95 error bar)")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "latency_per_stage.png", dpi=120)
    plt.close(fig)


def plot_grounding_ablation(summary_files: dict[str, dict], out: Path) -> None:
    if not summary_files:
        return
    labels, top1, top5, honesty, lat_p50, lat_p95 = [], [], [], [], [], []
    for path, s in summary_files.items():
        labels.append(s.get("config_name") or Path(path).parent.name)
        top1.append(s.get("grounding_top1", 0.0))
        top5.append(s.get("grounding_top5", 0.0))
        honesty.append(s.get("expected_failure_honesty", 0.0))
        lat_p50.append(s.get("latency_ms_p50", 0.0))
        lat_p95.append(s.get("latency_ms_p95", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = list(range(len(labels)))
    ax1.bar([xi - 0.25 for xi in x], top1, width=0.2, label="top-1")
    ax1.bar([xi - 0.05 for xi in x], top5, width=0.2, label="top-5")
    ax1.bar([xi + 0.20 for xi in x], honesty, width=0.2, label="honesty")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")
    ax1.set_ylabel("score [0, 1]")
    ax1.set_title("Grounding accuracy + expected-failure honesty")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize="small")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar([xi - 0.15 for xi in x], lat_p50, width=0.3, label="p50")
    ax2.bar([xi + 0.15 for xi in x], lat_p95, width=0.3, label="p95")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")
    ax2.set_ylabel("grounding latency (ms)")
    ax2.set_title("Grounding latency")
    ax2.legend(fontsize="small")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "grounding_ablation.png", dpi=120)
    plt.close(fig)


def plot_thermal(thermal_csv: Path, out: Path) -> None:
    if not thermal_csv.exists():
        return
    rows: list[dict[str, Any]] = []
    with thermal_csv.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return

    def _col(name: str) -> list[float]:
        vals = []
        for r in rows:
            v = r.get(name)
            if v in (None, "", "None"):
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(float("nan"))
        return vals

    t = _col("elapsed_s")
    hz = _col("detection_hz_recent")
    temp = _col("gpu_temp_c")
    util = _col("gpu_util_pct")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, hz, "g-", label="detection Hz")
    ax1.set_xlabel("elapsed (s)")
    ax1.set_ylabel("detection Hz", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, temp, "r-", label="GPU temp (°C)")
    ax2.plot(t, util, "b--", label="GPU util (%)", alpha=0.5)
    ax2.set_ylabel("GPU temp / util", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    lines = ax1.get_lines() + ax2.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax1.set_title("Jetson thermal soak")
    fig.tight_layout()
    fig.savefig(out / "thermal_plot.png", dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", default="eval/results")
    ap.add_argument("--thermal-csv", default="eval/results/thermal/thermal.csv")
    ap.add_argument("--out", default="eval/results/figures")
    args = ap.parse_args()

    root = Path(args.eval_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = _load_jsons(root, "summary.json")
    latencies = _load_jsons(root, "latencies_*.json")

    plot_grounding_ablation(summaries, out)
    plot_latency_per_stage(latencies, out)
    plot_thermal(Path(args.thermal_csv), out)

    produced = [p.name for p in out.iterdir() if p.suffix == ".png"]
    print(f"[figures] produced {len(produced)} figures in {out}:")
    for p in produced:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
