#!/usr/bin/env python3
"""Per-frame 2D grounding on real GO2 front-camera frames (no depth, no scene graph).

Runs the production detector -> segmenter -> encoder chain and the production grounding
score + rejection gate on each frame, for every query derived from the ground truth.
Thresholds and models are frozen from config/scene_profiles/dev_gpu.yaml; nothing here is
tuned. See docs/preregistration/2026-09-24-real-frames-2d-grounding.md.

Writes queries.csv (one row per frame x query) and latency.json into --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ros2_ws/src/go2_open_vocab_detector"))
sys.path.insert(0, str(REPO / "ros2_ws/src/go2_language_grounding"))

from go2_open_vocab_detector.backends import make_detector, make_encoder, make_segmenter  # noqa: E402
from go2_language_grounding.query_parser import parse  # noqa: E402
from go2_language_grounding.scoring import ScoreWeights, combine, cosine_similarity, lexical_label_match  # noqa: E402

# Frozen: detector default prompts (detector_node._DEFAULT_PROMPTS) and dev_gpu.yaml.
PROMPTS = ["chair", "sofa", "table", "desk", "window", "door", "person", "refrigerator",
           "television", "bed", "lamp", "potted plant", "bookshelf", "cup", "laptop", "backpack"]
OOV = ["guitar", "dog", "horse", "bicycle", "toilet"]
PROFILE = yaml.safe_load((REPO / "config/scene_profiles/dev_gpu.yaml").read_text())
DET = PROFILE["go2_open_vocab_detector"]["ros__parameters"]
GND = PROFILE["go2_language_grounding"]["ros__parameters"]
WEIGHTS = ScoreWeights()  # 0.7 / 0.2 / 0.1, grounding_node defaults


def score_candidates(parsed, labels, embs, encode_text):
    """Mirror of GroundingNode._score_candidates (grounding_node.py), spatial neutral 0.5."""
    attr_noun = (parsed.attribute + " " + parsed.target_noun).strip()
    plain_noun = parsed.target_noun.strip() or parsed.raw.strip()
    full = attr_noun or plain_noun
    full_emb = encode_text(full)
    plain_emb = encode_text(plain_noun) if parsed.attribute and plain_noun and plain_noun != full else None
    out = []
    for i, (label, emb) in enumerate(zip(labels, embs)):
        sim_full = cosine_similarity(emb, full_emb)
        if plain_emb is not None:
            delta = sim_full - cosine_similarity(emb, plain_emb)
            clip = 0.6 * sim_full + 0.4 * max(0.0, min(1.0, (delta + 0.05) / 0.15))
        else:
            clip = sim_full
        lab = lexical_label_match(parsed.target_noun, label)
        out.append((combine(clip, lab, 0.5, WEIGHTS), clip, lab, i))
    out.sort(key=lambda c: c[0], reverse=True)
    return out


def gate_refusal(top) -> str:
    """Mirror of the grounding node's v2 gate with reject_margin_min 0 (dev_gpu.yaml)."""
    score, clip, lab, _ = top
    if score < GND["reject_absolute_floor"]:
        return "below_absolute_floor"
    if lab < GND["reject_label_floor"] and clip < GND["reject_clip_floor"]:
        return "weak_label_and_clip"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="~/datasets/go2_real_frames_2d")
    ap.add_argument("--annotations", default=str(REPO / "eval/real_frames/annotations.yaml"))
    ap.add_argument("--out", default=str(REPO / "eval/results/real_frames_2d"))
    ap.add_argument("--positive-control", action="store_true", help="only run bus.jpg")
    args = ap.parse_args()

    device = DET["device"]
    det = make_detector(DET["detector_backend"])
    seg = make_segmenter(DET["segmenter_backend"])
    enc = make_encoder(DET["encoder_backend"])
    det.load(device=device, prompts=PROMPTS)
    seg.load(device=device)
    enc.load(device=device)
    text_cache: dict[str, np.ndarray] = {}

    def encode_text(t: str) -> np.ndarray:
        if t not in text_cache:
            text_cache[t] = enc.encode_text([t])[0]
        return text_cache[t]

    if args.positive_control:
        import ultralytics
        bus = Path(ultralytics.__file__).parent / "assets/bus.jpg"
        items = [{"frame": str(bus), "present": ["person"], "unsure": [], "bag": "pc", "block": "pc",
                  "viewpoint": "pc", "_absent": ["sofa"]}]
        frames_dir = Path("/")
    else:
        items = yaml.safe_load(Path(args.annotations).read_text())["frames"]
        frames_dir = Path(args.frames).expanduser()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, lat = [], {"detector": [], "segmenter": [], "encoder": [], "total": []}
    for k, fr in enumerate(items):
        img = cv2.imread(str(frames_dir / fr["frame"]))
        t0 = time.perf_counter()
        d = det.detect(img, float(DET["conf_threshold"]), int(DET["max_objects_per_frame"]))
        if len(d.labels):
            s = seg.segment(img, d.boxes_xyxy)
            e = enc.encode_images(img, d.boxes_xyxy, s.masks)
            embs, seg_ms, enc_ms = e.image_embeddings, s.latency_ms, e.latency_ms
        else:
            embs, seg_ms, enc_ms = np.zeros((0, enc.embedding_dim), np.float32), 0.0, 0.0
        total_ms = (time.perf_counter() - t0) * 1e3
        if k >= 3:  # skip warm-up frames for latency
            for key, v in (("detector", d.latency_ms), ("segmenter", seg_ms), ("encoder", enc_ms), ("total", total_ms)):
                lat[key].append(v)

        present, unsure = set(fr["present"]), set(fr["unsure"])
        absent = fr.get("_absent") or [c for c in PROMPTS + OOV if c not in present and c not in unsure]
        queries = [(c, "present") for c in sorted(present)] + [(c, "absent") for c in absent]
        for cls, truth in queries:
            q = f"go to the {cls}"
            parsed = parse(q)
            cands = score_candidates(parsed, d.labels, embs, encode_text) if len(d.labels) else []
            row = {"frame": fr["frame"], "bag": fr["bag"], "block": fr["block"], "viewpoint": fr["viewpoint"],
                   "query_class": cls, "in_vocab": cls in PROMPTS, "truth": truth, "query": q,
                   "parsed_noun": parsed.target_noun, "n_detections": len(d.labels)}
            if cands:
                top = cands[0]
                box = d.boxes_xyxy[top[3]]
                row.update(top_label=d.labels[top[3]], top_score=round(top[0], 4), top_clip=round(top[1], 4),
                           top_label_score=round(top[2], 4), top_det_conf=round(float(d.scores[top[3]]), 4),
                           top_box="%d %d %d %d" % tuple(int(v) for v in box),
                           gate_refusal=gate_refusal(top))
            else:
                row.update(top_label="", top_score="", top_clip="", top_label_score="", top_det_conf="",
                           top_box="", gate_refusal="no_detections")
            rows.append(row)
        print(f"[{k + 1}/{len(items)}] {fr['frame']}: {len(d.labels)} dets {sorted(set(d.labels))}", flush=True)

    name = "positive_control.csv" if args.positive_control else "queries.csv"
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if not args.positive_control:
        summ = {k: {"n": len(v), "p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95))}
                for k, v in lat.items() if v}
        summ["_note"] = "workstation GPU (dev_gpu profile), NOT Jetson; 960x540 input; first 3 frames excluded"
        (out / "latency.json").write_text(json.dumps(summ, indent=2))
    print(f"wrote {len(rows)} rows to {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
