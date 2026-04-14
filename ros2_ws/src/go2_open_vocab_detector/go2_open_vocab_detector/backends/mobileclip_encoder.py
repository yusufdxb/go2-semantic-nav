"""MobileCLIP image + text encoder (Apple ML Research).

MobileCLIP-S0/S2 matches OpenCLIP ViT-B/16 zero-shot accuracy at 3–5× the speed on
Jetson-class hardware. License is `apple-amlr` — research use OK; flag for any
commercial product deployment.

Loads via HuggingFace `timm` + `open_clip` hybrid path; the `mobileclip` pip
package is also supported if present.
"""

from __future__ import annotations

import time

import numpy as np

from .base import EncoderBackend, EncoderOutput

_MODEL_SPECS: dict[str, tuple[str, str, int]] = {
    # key → (open_clip arch, pretrained tag, embedding_dim)
    "mobileclip_s0": ("MobileCLIP-S0", "datacompdr", 512),
    "mobileclip_s1": ("MobileCLIP-S1", "datacompdr", 512),
    "mobileclip_s2": ("MobileCLIP-S2", "datacompdr", 512),
    "mobileclip_b": ("MobileCLIP-B", "datacompdr", 512),
}


class MobileClipEncoder(EncoderBackend):
    def __init__(self, name: str):
        if name not in _MODEL_SPECS:
            raise ValueError(f"Unknown MobileCLIP model: {name}")
        self.name = name
        self._arch, self._pretrained, self.embedding_dim = _MODEL_SPECS[name]
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._arch, pretrained=self._pretrained
        )
        model.to(device).eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._arch)
        self._device = device
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 256, 256), device=device)
            _ = model.encode_image(dummy)

    def encode_images(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        masks: np.ndarray | None,
    ) -> EncoderOutput:
        import torch
        from PIL import Image

        if self._model is None:
            raise RuntimeError(f"{self.name}: load() must be called before encode_images()")
        start_ns = time.perf_counter_ns()

        n = int(boxes_xyxy.shape[0])
        if n == 0:
            return EncoderOutput(
                image_embeddings=np.zeros((0, self.embedding_dim), dtype=np.float32),
                latency_ms=(time.perf_counter_ns() - start_ns) / 1e6,
            )

        image_rgb = image_bgr[:, :, ::-1]
        pil_crops = []
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image_rgb.shape[1], x2)
            y2 = min(image_rgb.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((1, 1, 3), dtype=np.uint8)
            else:
                crop = image_rgb[y1:y2, x1:x2].copy()
                if masks is not None and masks.shape[0] > i:
                    mask = masks[i, y1:y2, x1:x2]
                    if mask.shape == crop.shape[:2]:
                        crop = crop * mask[..., None]
            pil_crops.append(Image.fromarray(crop))

        batch = torch.stack([self._preprocess(c) for c in pil_crops]).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return EncoderOutput(
            image_embeddings=feats.detach().cpu().numpy().astype(np.float32),
            latency_ms=(time.perf_counter_ns() - start_ns) / 1e6,
        )

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"{self.name}: load() must be called before encode_text()")
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats.detach().cpu().numpy().astype(np.float32)
