"""SAM 2 segmenter with memory-bank disabled (single-frame mode)."""

from __future__ import annotations

import time

import numpy as np

from .base import SegmenterBackend, SegmenterOutput

_MODEL_SPECS: dict[str, tuple[str, str]] = {
    # key → (config yaml, checkpoint name)
    "sam2_tiny": ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "sam2_small": ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "sam2_base_plus": ("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
}


class Sam2Segmenter(SegmenterBackend):
    def __init__(self, name: str):
        if name not in _MODEL_SPECS:
            raise ValueError(f"Unknown SAM 2 model: {name}")
        self.name = name
        self._cfg_yaml, self._ckpt_name = _MODEL_SPECS[name]
        self._predictor = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 requires the `sam-2` package (pip install from "
                "https://github.com/facebookresearch/sam2). Not yet installed."
            ) from exc

        import os

        # Config is loaded from sam2's packaged assets; checkpoint path comes from env.
        ckpt = os.environ.get(
            f"SAM2_CHECKPOINT_{self.name.upper()}",
            os.path.expanduser(f"~/.cache/sam2/{self._ckpt_name}"),
        )
        sam = build_sam2(self._cfg_yaml, ckpt, device=device)
        self._predictor = SAM2ImagePredictor(sam)
        self._device = device

    def segment(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> SegmenterOutput:
        if self._predictor is None:
            raise RuntimeError(f"{self.name}: load() must be called before segment()")
        start_ns = time.perf_counter_ns()

        image_rgb = image_bgr[:, :, ::-1]
        self._predictor.set_image(image_rgb)

        h, w = image_bgr.shape[:2]
        n = int(boxes_xyxy.shape[0])
        if n == 0:
            return SegmenterOutput(masks=np.zeros((0, h, w), dtype=bool), latency_ms=0.0)

        masks = np.zeros((n, h, w), dtype=bool)
        for i in range(n):
            pred, _, _ = self._predictor.predict(box=boxes_xyxy[i], multimask_output=False)
            masks[i] = pred[0].astype(bool)
        return SegmenterOutput(
            masks=masks,
            latency_ms=(time.perf_counter_ns() - start_ns) / 1e6,
        )
