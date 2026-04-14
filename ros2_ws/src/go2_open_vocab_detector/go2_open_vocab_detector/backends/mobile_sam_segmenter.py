"""MobileSAM-backed segmenter: image encoder + box-prompted decoder."""

from __future__ import annotations

import time

import numpy as np

from .base import SegmenterBackend, SegmenterOutput


class MobileSamSegmenter(SegmenterBackend):
    """Wraps the `mobile_sam` pip package. Dev-only; on Jetson use NanoSamSegmenter.

    One image-encoder forward per frame; one decoder forward per box prompt.
    """

    def __init__(self, name: str = "mobile_sam", checkpoint: str | None = None):
        self.name = name
        self._checkpoint = checkpoint
        self._predictor = None
        self._device: str | None = None

    def _resolve_checkpoint(self) -> str:
        import os

        if self._checkpoint and os.path.exists(self._checkpoint):
            return self._checkpoint
        for cand in (
            self._checkpoint or "",
            os.path.expanduser("~/.cache/mobile_sam/mobile_sam.pt"),
            os.path.join(os.getcwd(), "mobile_sam.pt"),
            os.path.join(os.getcwd(), "models", "mobile_sam.pt"),
        ):
            if cand and os.path.exists(cand):
                return cand
        raise FileNotFoundError(
            "MobileSAM checkpoint not found. Run `python scripts/prefetch_models.py --backends mobile_sam` "
            "or place mobile_sam.pt at ~/.cache/mobile_sam/mobile_sam.pt"
        )

    def load(self, device: str) -> None:
        # Import lazily.
        from mobile_sam import SamPredictor, sam_model_registry

        ckpt = self._resolve_checkpoint()
        sam = sam_model_registry["vit_t"](checkpoint=ckpt)
        sam.to(device=device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self._device = device

    def segment(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> SegmenterOutput:
        if self._predictor is None:
            raise RuntimeError("MobileSAM: load() must be called before segment()")
        start_ns = time.perf_counter_ns()

        # MobileSAM expects RGB.
        image_rgb = image_bgr[:, :, ::-1]
        self._predictor.set_image(image_rgb)

        n = int(boxes_xyxy.shape[0])
        if n == 0:
            return SegmenterOutput(
                masks=np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool),
                latency_ms=(time.perf_counter_ns() - start_ns) / 1e6,
            )

        masks = np.zeros((n, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool)
        for i in range(n):
            box = boxes_xyxy[i]
            pred, _, _ = self._predictor.predict(
                box=box.astype(np.float32),
                multimask_output=False,
            )
            # pred shape: (1, H, W) bool/uint8
            masks[i] = pred[0].astype(bool)

        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return SegmenterOutput(masks=masks, latency_ms=latency_ms)
