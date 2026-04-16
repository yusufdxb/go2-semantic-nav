"""Grounding-DINO open-vocabulary detector — offboard / RTX only.

Needs the `groundingdino-py` pip package plus its `GroundingDINO_SwinT_OGC.py`
config + `groundingdino_swint_ogc.pth` weights. Not recommended for Jetson
because the required MultiScaleDeformableAttn TRT plugin is not in stock L4T TRT.
"""

from __future__ import annotations

import time

import numpy as np

from .base import DetectorBackend, DetectorOutput


class GroundingDinoDetector(DetectorBackend):
    def __init__(self, name: str = "grounding_dino_tiny"):
        self.name = name
        self._model = None
        self._device: str | None = None
        self._prompts: list[str] = []
        self._hf_id = (
            "IDEA-Research/grounding-dino-tiny"
            if name == "grounding_dino_tiny"
            else "IDEA-Research/grounding-dino-base"
        )

    def load(self, device: str, prompts: list[str]) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._hf_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self._hf_id).to(device).eval()
        self._device = device
        self._prompts = list(prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        self._prompts = list(prompts)

    def detect(self, image_bgr: np.ndarray, conf_threshold: float, max_objects: int) -> DetectorOutput:
        import torch
        from PIL import Image

        if self._model is None:
            raise RuntimeError(f"{self.name}: load() must be called before detect()")
        start_ns = time.perf_counter_ns()

        text = ". ".join(self._prompts) + "."
        image_rgb = image_bgr[:, :, ::-1]
        pil = Image.fromarray(image_rgb)
        inputs = self._processor(images=pil, text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]], device=self._device)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=conf_threshold,
            text_threshold=conf_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        labels = [str(lbl) for lbl in results.get("labels", results.get("text_labels", []))]

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
