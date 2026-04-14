"""SigLIP image + text encoder (HuggingFace transformers)."""

from __future__ import annotations

import time

import numpy as np

from .base import EncoderBackend, EncoderOutput

_MODEL_SPECS: dict[str, tuple[str, int]] = {
    "siglip_base": ("google/siglip-base-patch16-224", 768),
    "siglip_so400m": ("google/siglip-so400m-patch14-384", 1152),
}


class SigLipEncoder(EncoderBackend):
    def __init__(self, name: str):
        if name not in _MODEL_SPECS:
            raise ValueError(f"Unknown SigLIP model: {name}")
        self.name = name
        self._hf_id, self.embedding_dim = _MODEL_SPECS[name]
        self._model = None
        self._processor = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        from transformers import AutoModel, AutoProcessor

        self._model = AutoModel.from_pretrained(self._hf_id).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(self._hf_id)
        self._device = device

    def encode_images(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        masks: np.ndarray | None,
    ) -> EncoderOutput:
        import torch
        from PIL import Image

        if self._model is None or self._processor is None:
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

        inputs = self._processor(images=pil_crops, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return EncoderOutput(
            image_embeddings=feats.detach().cpu().numpy().astype(np.float32),
            latency_ms=(time.perf_counter_ns() - start_ns) / 1e6,
        )

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError(f"{self.name}: load() must be called before encode_text()")
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats.detach().cpu().numpy().astype(np.float32)
