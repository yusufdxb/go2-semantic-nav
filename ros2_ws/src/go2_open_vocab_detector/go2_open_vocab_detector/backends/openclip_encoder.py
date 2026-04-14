"""OpenCLIP-backed per-object image encoder + text encoder."""

from __future__ import annotations

import time

import numpy as np

from .base import EncoderBackend, EncoderOutput

_MODEL_SPECS: dict[str, tuple[str, str, int]] = {
    # key → (open_clip arch, pretrained tag, embedding_dim)
    "openclip_vit_b16": ("ViT-B-16", "laion2b_s34b_b88k", 512),
    "openclip_vit_l14": ("ViT-L-14", "laion2b_s32b_b82k", 768),
    "openclip_vit_h14": ("ViT-H-14", "laion2b_s32b_b79k", 1024),
}


class OpenClipEncoder(EncoderBackend):
    def __init__(self, name: str):
        if name not in _MODEL_SPECS:
            raise ValueError(f"Unknown OpenCLIP model: {name}")
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
        # Warmup
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=device)
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
        pil_crops: list = []
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_rgb.shape[1], x2)
            y2 = min(image_rgb.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                # Degenerate box; use a single-pixel crop to keep tensor shapes aligned.
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
        embeddings = feats.detach().cpu().numpy().astype(np.float32)

        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return EncoderOutput(image_embeddings=embeddings, latency_ms=latency_ms)

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
