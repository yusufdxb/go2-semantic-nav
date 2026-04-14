#!/usr/bin/env python3
"""Export YOLO-World v2 to ONNX → TensorRT FP16 for Jetson deployment.

Two-step workflow:
  1. Re-parameterize with a fixed prompt vocabulary (avoids dynamic text-token branch).
  2. Export to ONNX via ultralytics, then build a TensorRT engine via `trtexec`.

Run on the target Jetson for best results (engines are device/TRT-version specific).
On an x86 dev host this script still produces the ONNX but `trtexec` invocation is
skipped unless --trtexec is explicitly pointed at an installed binary.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--prompts", required=True, help="YAML with a `prompts:` list")
    ap.add_argument("--out-onnx", default="models/yolo_world_v2_s.onnx")
    ap.add_argument("--out-engine", default="models/yolo_world_v2_s.engine")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--int8", action="store_true", default=False)
    ap.add_argument("--trtexec", default=shutil.which("trtexec") or "")
    args = ap.parse_args()

    try:
        import yaml
    except ImportError:
        print("pyyaml required: pip install pyyaml", file=sys.stderr)
        return 1
    with open(args.prompts) as f:
        cfg = yaml.safe_load(f)
    prompts = [str(p) for p in cfg.get("prompts", [])]
    if not prompts:
        print(f"no prompts in {args.prompts}", file=sys.stderr)
        return 1
    print(f"[export] prompts ({len(prompts)}): {prompts[:6]}...")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.set_classes(prompts)

    out_dir = Path(args.out_onnx).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # ultralytics auto-saves alongside the .pt; rename after.
    onnx_path = model.export(format="onnx", imgsz=args.img_size, opset=17, dynamic=False, simplify=True)
    shutil.move(str(onnx_path), args.out_onnx)
    print(f"[export] ONNX → {args.out_onnx}")

    if not args.trtexec:
        print("[export] `trtexec` not found. On the Jetson run:")
        print(
            f"  trtexec --onnx={args.out_onnx} --saveEngine={args.out_engine} "
            f"--{'fp16' if args.fp16 else 'fp32'} --workspace=4096"
        )
        return 0

    cmd = [
        args.trtexec,
        f"--onnx={args.out_onnx}",
        f"--saveEngine={args.out_engine}",
        "--workspace=4096",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.int8:
        cmd.append("--int8")
    print("[export] running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[export] TRT engine → {args.out_engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
