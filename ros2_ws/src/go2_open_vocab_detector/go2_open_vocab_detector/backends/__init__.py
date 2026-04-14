"""Pluggable detector / segmenter / encoder backends."""

from .base import (
    DetectorBackend,
    DetectorOutput,
    EncoderBackend,
    EncoderOutput,
    SegmenterBackend,
    SegmenterOutput,
)
from .factory import make_detector, make_encoder, make_segmenter

__all__ = [
    "DetectorBackend",
    "DetectorOutput",
    "EncoderBackend",
    "EncoderOutput",
    "SegmenterBackend",
    "SegmenterOutput",
    "make_detector",
    "make_encoder",
    "make_segmenter",
]
