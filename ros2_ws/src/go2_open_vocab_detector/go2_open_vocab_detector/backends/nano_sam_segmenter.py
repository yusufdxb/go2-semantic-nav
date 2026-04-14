"""NanoSAM-backed segmenter — Jetson-native TensorRT path.

NanoSAM (NVIDIA-AI-IOT/nanosam) distills MobileSAM onto a ResNet-18 image encoder,
ships pre-built TRT engines for the encoder + MobileSAM decoder, and hits ~25 ms
end-to-end on a Jetson Orin NX 25 W after the one-time engine build.

This backend is *Jetson-only*. It imports `nanosam` lazily; dev machines that don't
have the JetPack TRT bindings will see a clean ImportError at load() time.
"""

from __future__ import annotations

import time

import numpy as np

from .base import SegmenterBackend, SegmenterOutput


class NanoSamSegmenter(SegmenterBackend):
    def __init__(self, name: str = "nano_sam"):
        self.name = name
        self._predictor = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        try:
            from nanosam.utils.predictor import Predictor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "NanoSAM is Jetson-only. Install NVIDIA-AI-IOT/nanosam and its TRT engine files, "
                "or switch to `segmenter_backend: mobile_sam` on dev hosts."
            ) from exc

        # Engine paths follow the NanoSAM README defaults. Users override via env vars
        # before launch if their engines live elsewhere.
        import os

        encoder_engine = os.environ.get(
            "NANOSAM_ENCODER_ENGINE",
            os.path.expanduser("~/nanosam/data/resnet18_image_encoder.engine"),
        )
        decoder_engine = os.environ.get(
            "NANOSAM_DECODER_ENGINE",
            os.path.expanduser("~/nanosam/data/mobile_sam_mask_decoder.engine"),
        )
        self._predictor = Predictor(encoder_engine, decoder_engine)
        self._device = device

    def segment(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> SegmenterOutput:
        if self._predictor is None:
            raise RuntimeError("NanoSAM: load() must be called before segment()")
        from PIL import Image as PILImage

        start_ns = time.perf_counter_ns()
        image_rgb = image_bgr[:, :, ::-1]
        pil = PILImage.fromarray(image_rgb)
        self._predictor.set_image(pil)

        h, w = image_bgr.shape[:2]
        n = int(boxes_xyxy.shape[0])
        masks = np.zeros((n, h, w), dtype=bool)
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(float)
            # NanoSAM uses (x, y) point prompts with positive labels; give the 4 corners
            # as positive to emulate box prompting. The shipped predictor also accepts
            # box-as-two-points on modern versions.
            points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            point_labels = np.array([2, 3], dtype=np.int32)  # top-left, bottom-right markers
            mask, _, _ = self._predictor.predict(points, point_labels)
            masks[i] = mask[0].astype(bool)

        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return SegmenterOutput(masks=masks, latency_ms=latency_ms)
