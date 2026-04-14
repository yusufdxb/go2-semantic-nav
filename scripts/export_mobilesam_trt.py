#!/usr/bin/env python3
"""Export MobileSAM image encoder + mask decoder to ONNX → TensorRT FP16.

On Jetson, prefer the pre-distilled NanoSAM pipeline (see
https://github.com/NVIDIA-AI-IOT/nanosam) — it ships ready-to-use TRT engines.
This script is a fallback for users who already have the MobileSAM pt file and
want bespoke ONNX engines.
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
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/.cache/mobile_sam/mobile_sam.pt"))
    ap.add_argument("--out-encoder-onnx", default="models/mobilesam_encoder.onnx")
    ap.add_argument("--out-decoder-onnx", default="models/mobilesam_decoder.onnx")
    ap.add_argument("--out-encoder-engine", default="models/mobilesam_encoder.engine")
    ap.add_argument("--out-decoder-engine", default="models/mobilesam_decoder.engine")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--trtexec", default=shutil.which("trtexec") or "")
    args = ap.parse_args()

    try:
        import torch
        from mobile_sam import sam_model_registry
        from mobile_sam.utils.onnx import SamOnnxModel
    except ImportError as exc:
        print(f"required: pip install mobile-sam ({exc})", file=sys.stderr)
        return 1

    sam = sam_model_registry["vit_t"](checkpoint=args.checkpoint)
    sam.eval()

    # --- Image encoder (tiny_vit) ---
    encoder = sam.image_encoder
    dummy = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    out_dir = Path(args.out_encoder_onnx).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder,
        dummy,
        args.out_encoder_onnx,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=17,
    )
    print(f"[export] encoder ONNX → {args.out_encoder_onnx}")

    # --- Mask decoder (box/point → mask) ---
    decoder = SamOnnxModel(sam, return_single_mask=True)
    dummy_embed = torch.zeros((1, 256, 64, 64), dtype=torch.float32)
    dummy_points = torch.zeros((1, 2, 2), dtype=torch.float32)
    dummy_labels = torch.ones((1, 2), dtype=torch.float32)
    dummy_mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    dummy_has_mask = torch.tensor([0], dtype=torch.float32)
    dummy_orig_size = torch.tensor([1024, 1024], dtype=torch.float32)

    torch.onnx.export(
        decoder,
        (
            dummy_embed,
            dummy_points,
            dummy_labels,
            dummy_mask_input,
            dummy_has_mask,
            dummy_orig_size,
        ),
        args.out_decoder_onnx,
        input_names=[
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
        ],
        output_names=["masks", "iou_predictions", "low_res_masks"],
        opset_version=17,
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
    )
    print(f"[export] decoder ONNX → {args.out_decoder_onnx}")

    if not args.trtexec:
        print(
            "[export] `trtexec` not found. On the Jetson run (encoder then decoder):\n"
            f"  trtexec --onnx={args.out_encoder_onnx} --saveEngine={args.out_encoder_engine} "
            f"--{'fp16' if args.fp16 else 'fp32'} --workspace=4096\n"
            f"  trtexec --onnx={args.out_decoder_onnx} --saveEngine={args.out_decoder_engine} "
            f"--{'fp16' if args.fp16 else 'fp32'} --workspace=1024"
        )
        return 0

    for onnx_p, engine_p, ws in (
        (args.out_encoder_onnx, args.out_encoder_engine, 4096),
        (args.out_decoder_onnx, args.out_decoder_engine, 1024),
    ):
        cmd = [args.trtexec, f"--onnx={onnx_p}", f"--saveEngine={engine_p}", f"--workspace={ws}"]
        if args.fp16:
            cmd.append("--fp16")
        print("[export] running:", " ".join(cmd))
        subprocess.check_call(cmd)
        print(f"[export] TRT engine → {engine_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
