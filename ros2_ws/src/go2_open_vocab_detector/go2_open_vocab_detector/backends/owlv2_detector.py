"""OWLv2 open-vocabulary detector (HuggingFace transformers).

Used as a burst-mode fallback when YOLO-World does not match the prompt vocabulary.
Slow (~150 ms on Orin NX with NanoOWLv2 TRT; ~3 s with stock HF on the same device).
"""

from __future__ import annotations

import time

import numpy as np

from .base import DetectorBackend, DetectorOutput

_MODEL_SPECS: dict[str, str] = {
    "owlv2_base": "google/owlv2-base-patch16-ensemble",
    "owlv2_large": "google/owlv2-large-patch14-ensemble",
}


class OwlV2Detector(DetectorBackend):
    def __init__(self, name: str):
        if name not in _MODEL_SPECS:
            raise ValueError(f"Unknown OWLv2 model: {name}")
        self.name = name
        self._hf_id = _MODEL_SPECS[name]
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._prompts: list[str] = []

    def load(self, device: str, prompts: list[str]) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._processor = Owlv2Processor.from_pretrained(self._hf_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(self._hf_id).to(device).eval()
        self._device = device
        self._prompts = list(prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        self._prompts = list(prompts)

    def detect(self, image_bgr: np.ndarray, conf_threshold: float, max_objects: int) -> DetectorOutput:
        import torch
        from PIL import Image

        if self._model is None or self._processor is None:
            raise RuntimeError(f"{self.name}: load() must be called before detect()")
        if not self._prompts:
            return DetectorOutput(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=[],
                latency_ms=0.0,
            )
        start_ns = time.perf_counter_ns()

        image_rgb = image_bgr[:, :, ::-1]
        pil = Image.fromarray(image_rgb)
        texts = [[f"a photo of a {p}" for p in self._prompts]]
        inputs = self._processor(text=texts, images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        label_idx = results["labels"].detach().cpu().numpy()
        labels = [self._prompts[i] if i < len(self._prompts) else f"class_{i}" for i in label_idx]

        # Cap max objects
        if len(labels) > max_objects:
            order = np.argsort(-scores)[:max_objects]
            boxes = boxes[order]
            scores = scores[order]
            labels = [labels[i] for i in order]

        return DetectorOutput(
            boxes_xyxy=boxes,
            scores=scores,
            labels=labels,
            latency_ms=(time.perf_counter_ns() - start_ns) / 1e6,
        )
