#!/usr/bin/env python3
"""Pre-download model weights so offline Jetson deployments don't hit the network.

Usage:
    python scripts/prefetch_models.py --backends yolo_world_v2_s,mobile_sam,openclip_vit_b16
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable


def _fetch_yolo_world_v2_s() -> None:
    from ultralytics import YOLO

    print("[prefetch] YOLO-World v2-s")
    YOLO("yolov8s-worldv2.pt")


def _fetch_yolo_world_v2_m() -> None:
    from ultralytics import YOLO

    print("[prefetch] YOLO-World v2-m")
    YOLO("yolov8m-worldv2.pt")


def _fetch_yoloe_11s() -> None:
    from ultralytics import YOLO

    print("[prefetch] YOLOE-11s")
    YOLO("yoloe-11s-seg.pt")


def _fetch_mobile_sam() -> None:
    # mobile_sam checkpoint is not auto-downloaded by the pip package.
    # Fetch from the upstream release URL.
    import os
    import urllib.request

    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    dst = os.path.expanduser("~/.cache/mobile_sam/mobile_sam.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"[prefetch] MobileSAM → {dst}")
        urllib.request.urlretrieve(url, dst)
    else:
        print(f"[prefetch] MobileSAM already cached at {dst}")


def _fetch_openclip_vit_b16() -> None:
    import open_clip

    print("[prefetch] OpenCLIP ViT-B/16 (laion2b_s34b_b88k)")
    open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")


def _fetch_openclip_vit_l14() -> None:
    import open_clip

    print("[prefetch] OpenCLIP ViT-L/14 (laion2b_s32b_b82k)")
    open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")


FETCHERS: dict[str, Callable[[], None]] = {
    "yolo_world_v2_s": _fetch_yolo_world_v2_s,
    "yolo_world_v2_m": _fetch_yolo_world_v2_m,
    "yoloe_11s": _fetch_yoloe_11s,
    "mobile_sam": _fetch_mobile_sam,
    "openclip_vit_b16": _fetch_openclip_vit_b16,
    "openclip_vit_l14": _fetch_openclip_vit_l14,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backends",
        default="yolo_world_v2_s,mobile_sam,openclip_vit_b16",
        help=f"Comma-separated backend ids. Known: {', '.join(FETCHERS)}",
    )
    args = ap.parse_args()

    failed = []
    for name in [n.strip() for n in args.backends.split(",") if n.strip()]:
        if name not in FETCHERS:
            print(f"[prefetch] UNKNOWN backend: {name}")
            failed.append(name)
            continue
        try:
            FETCHERS[name]()
        except Exception as exc:  # continue on failure; best-effort
            print(f"[prefetch] FAILED {name}: {exc}")
            failed.append(name)

    if failed:
        print(f"[prefetch] FAILED: {failed}")
        return 1
    print("[prefetch] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
