"""EfficientSAM segmenter. Lazy-import so missing deps fail cleanly at load()."""

from __future__ import annotations

import time

import numpy as np

from .base import SegmenterBackend, SegmenterOutput


class EfficientSamSegmenter(SegmenterBackend):
    def __init__(self, name: str):
        self.name = name
        self._size = "ti" if "ti" in name else "s"
        self._model = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        try:
            from efficient_sam.build_efficient_sam import (  # type: ignore
                build_efficient_sam_vits,
                build_efficient_sam_vitt,
            )
        except ImportError as exc:
            raise RuntimeError(
                "EfficientSAM requires the `efficient_sam` package (pip install from "
                "https://github.com/yformer/EfficientSAM). Not yet installed."
            ) from exc

        builder = build_efficient_sam_vitt if self._size == "ti" else build_efficient_sam_vits
        self._model = builder().to(device).eval()
        self._device = device

    def segment(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> SegmenterOutput:
        import torch

        if self._model is None:
            raise RuntimeError(f"{self.name}: load() must be called before segment()")
        start_ns = time.perf_counter_ns()

        h, w = image_bgr.shape[:2]
        n = int(boxes_xyxy.shape[0])
        if n == 0:
            return SegmenterOutput(masks=np.zeros((0, h, w), dtype=bool), latency_ms=0.0)

        # EfficientSAM expects (B, C, H, W) float in [0,1] RGB.
        image_rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        img_t = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)

        # Box as two points (top-left, bottom-right). Labels: 2=top-left, 3=bottom-right.
        pts = torch.zeros((1, n, 2, 2), device=self._device)
        lbls = torch.zeros((1, n, 2), dtype=torch.long, device=self._device)
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i]
            pts[0, i, 0] = torch.tensor([x1, y1])
            pts[0, i, 1] = torch.tensor([x2, y2])
            lbls[0, i, 0] = 2
            lbls[0, i, 1] = 3

        with torch.no_grad():
            masks_t, _, _ = self._model(img_t, pts, lbls)
            # masks_t: (1, n, num_multimask, H, W)
            best = masks_t[0, :, 0, :, :]
            bin_masks = (best > 0).detach().cpu().numpy()

        return SegmenterOutput(
            masks=bin_masks.astype(bool),
            latency_ms=(time.perf_counter_ns() - start_ns) / 1e6,
        )
