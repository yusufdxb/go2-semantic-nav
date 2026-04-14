"""Ultralytics-backed open-vocabulary detector (YOLO-World v2 and YOLOE)."""

from __future__ import annotations

import time

import numpy as np

from .base import DetectorBackend, DetectorOutput

_MODEL_NAMES: dict[str, str] = {
    "yolo_world_v2_s": "yolov8s-worldv2.pt",
    "yolo_world_v2_m": "yolov8m-worldv2.pt",
    "yolo_world_v2_l": "yolov8l-worldv2.pt",
    "yolo_world_v2_x": "yolov8x-worldv2.pt",
    "yoloe_11s": "yoloe-11s-seg.pt",
}


class UltralyticsOpenVocabDetector(DetectorBackend):
    """YOLO-World v2 / YOLOE via the ultralytics API.

    Design notes:
    - set_prompts() calls `model.set_classes(prompts)` which re-parameterizes the
      text-prompt branch; for deployment, export AFTER calling this so TRT engines
      bake the vocab in.
    - YOLOE supports both prompt-free and prompt-driven modes. When ``name`` starts
      with ``yoloe`` and ``prompts`` is empty, we use the prompt-free mode.
    """

    def __init__(self, name: str):
        if name not in _MODEL_NAMES:
            raise ValueError(f"Unknown ultralytics model: {name}")
        self.name = name
        self._weight_file = _MODEL_NAMES[name]
        self._model = None
        self._device: str | None = None
        self._current_prompts: list[str] = []

    def load(self, device: str, prompts: list[str]) -> None:
        from ultralytics import YOLO  # imported lazily; heavy import

        self._model = YOLO(self._weight_file)
        self._device = device
        if prompts:
            self.set_prompts(prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        if self._model is None:
            raise RuntimeError(f"{self.name}: load() must be called before set_prompts()")
        if self.name.startswith("yolo_world") and prompts:
            self._model.set_classes(prompts)
        elif self.name.startswith("yoloe") and prompts:
            # YOLOE accepts text prompts via the same API in ultralytics 8.4+.
            self._model.set_classes(prompts)
        self._current_prompts = list(prompts)

    def detect(self, image_bgr: np.ndarray, conf_threshold: float, max_objects: int) -> DetectorOutput:
        if self._model is None:
            raise RuntimeError(f"{self.name}: load() must be called before detect()")
        start_ns = time.perf_counter_ns()

        results = self._model.predict(
            source=image_bgr,
            conf=conf_threshold,
            verbose=False,
            device=self._device,
            max_det=max_objects,
        )
        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6

        if not results:
            return DetectorOutput(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=[],
                latency_ms=latency_ms,
            )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return DetectorOutput(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=[],
                latency_ms=latency_ms,
            )

        boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls_idx = r.boxes.cls.detach().cpu().numpy().astype(int)

        # Prefer r.names (ultralytics standard) which is a dict[int, str].
        name_map = r.names if hasattr(r, "names") and r.names else {}
        labels: list[str] = []
        for c in cls_idx.tolist():
            label = name_map.get(c, f"class_{c}") if isinstance(name_map, dict) else f"class_{c}"
            labels.append(str(label))

        return DetectorOutput(
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            labels=labels,
            latency_ms=latency_ms,
        )
