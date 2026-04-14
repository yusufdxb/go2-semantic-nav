"""Abstract backend interfaces for detection, segmentation, and embedding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorOutput:
    """Open-vocabulary detections for a single image."""

    boxes_xyxy: np.ndarray  # (N, 4) float32
    scores: np.ndarray      # (N,)  float32
    labels: list[str]       # length N
    latency_ms: float


@dataclass
class SegmenterOutput:
    """Masks for a set of box prompts over a single image."""

    masks: np.ndarray       # (N, H, W) bool
    latency_ms: float


@dataclass
class EncoderOutput:
    """Image embeddings per region and optional text embeddings."""

    image_embeddings: np.ndarray  # (N, D) float32, L2-normalized
    latency_ms: float


class DetectorBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def load(self, device: str, prompts: list[str]) -> None:
        """Load weights and warm up on the given device. Called from on_configure."""

    @abstractmethod
    def set_prompts(self, prompts: list[str]) -> None:
        """Update the open-vocab prompt set. May be called at runtime."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, conf_threshold: float, max_objects: int) -> DetectorOutput:
        """Run inference on an RGB/BGR uint8 image and return boxes + labels."""


class SegmenterBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def segment(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> SegmenterOutput: ...


class EncoderBackend(ABC):
    name: str = "abstract"
    embedding_dim: int = 0

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def encode_images(self, image_bgr: np.ndarray, boxes_xyxy: np.ndarray, masks: np.ndarray | None) -> EncoderOutput:
        """Return unit-normalized embeddings per box (masked crops preferred when masks is not None)."""

    @abstractmethod
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Return unit-normalized text embeddings of shape (len(texts), D)."""
