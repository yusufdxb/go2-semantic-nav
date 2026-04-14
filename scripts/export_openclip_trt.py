#!/usr/bin/env python3
"""Export OpenCLIP image encoder to ONNX → TensorRT FP16.

Only the image tower is exported here; the text tower is small enough to stay on
PyTorch at query time, or be exported separately.

Known gotcha: OpenCLIP ViT-L/14 on TRT 10.x sometimes runs slower in FP16 than
FP32 due to GELU precision fallback. Pass --obey-precision to force FP16.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--out-onnx", default="models/openclip_vit_b16_image.onnx")
    ap.add_argument("--out-engine", default="models/openclip_vit_b16_image.engine")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--obey-precision", action="store_true", default=False)
    ap.add_argument("--trtexec", default=shutil.which("trtexec") or "")
    args = ap.parse_args()

    try:
        import open_clip
        import torch
    except ImportError as exc:
        print(f"required: pip install open_clip_torch torch ({exc})", file=sys.stderr)
        return 1

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.eval()

    class ImageEncoderWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, images):
            feats = self.m.encode_image(images)
            return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    wrapper = ImageEncoderWrapper(model)
    dummy = torch.zeros((1, 3, args.img_size, args.img_size), dtype=torch.float32)

    out_dir = Path(args.out_onnx).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out_onnx,
        input_names=["images"],
        output_names=["image_embeddings"],
        opset_version=17,
        dynamic_axes={
            "images": {0: "batch"},
            "image_embeddings": {0: "batch"},
        },
    )
    print(f"[export] ONNX → {args.out_onnx}")

    if not args.trtexec:
        print("[export] `trtexec` not found. On the Jetson run:")
        print(
            f"  trtexec --onnx={args.out_onnx} --saveEngine={args.out_engine} "
            f"--{'fp16' if args.fp16 else 'fp32'} --workspace=4096 "
            f"--minShapes=images:1x3x{args.img_size}x{args.img_size} "
            f"--optShapes=images:16x3x{args.img_size}x{args.img_size} "
            f"--maxShapes=images:32x3x{args.img_size}x{args.img_size}"
        )
        return 0

    cmd = [
        args.trtexec,
        f"--onnx={args.out_onnx}",
        f"--saveEngine={args.out_engine}",
        "--workspace=4096",
        f"--minShapes=images:1x3x{args.img_size}x{args.img_size}",
        f"--optShapes=images:16x3x{args.img_size}x{args.img_size}",
        f"--maxShapes=images:32x3x{args.img_size}x{args.img_size}",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.obey_precision:
        cmd.append("--precisionConstraints=obey")
        cmd.append("--layerPrecisions=*:fp16")
    print("[export] running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[export] TRT engine → {args.out_engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
