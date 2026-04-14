"""Factory functions that instantiate backends by string identifier.

These are lightweight indirections so node parameters can swap implementations
without changing node code. Each factory imports its module lazily to avoid
loading ML frameworks at ROS discovery time.
"""

from __future__ import annotations

from .base import DetectorBackend, EncoderBackend, SegmenterBackend


def make_detector(name: str) -> DetectorBackend:
    if name in {"yolo_world_v2_s", "yolo_world_v2_m", "yolo_world_v2_l", "yolo_world_v2_x", "yoloe_11s"}:
        from .ultralytics_detector import UltralyticsOpenVocabDetector

        return UltralyticsOpenVocabDetector(name)
    if name == "owlv2_base":
        from .owlv2_detector import OwlV2Detector

        return OwlV2Detector(name)
    if name == "grounding_dino_tiny":
        from .grounding_dino_detector import GroundingDinoDetector

        return GroundingDinoDetector(name)
    raise ValueError(f"Unknown detector backend: {name}")


def make_segmenter(name: str) -> SegmenterBackend:
    if name == "mobile_sam":
        from .mobile_sam_segmenter import MobileSamSegmenter

        return MobileSamSegmenter(name)
    if name == "nano_sam":
        from .nano_sam_segmenter import NanoSamSegmenter

        return NanoSamSegmenter(name)
    if name in {"efficient_sam_ti", "efficient_sam_s"}:
        from .efficient_sam_segmenter import EfficientSamSegmenter

        return EfficientSamSegmenter(name)
    if name in {"sam2_tiny", "sam2_small", "sam2_base_plus"}:
        from .sam2_segmenter import Sam2Segmenter

        return Sam2Segmenter(name)
    raise ValueError(f"Unknown segmenter backend: {name}")


def make_encoder(name: str) -> EncoderBackend:
    if name in {"openclip_vit_b16", "openclip_vit_l14", "openclip_vit_h14"}:
        from .openclip_encoder import OpenClipEncoder

        return OpenClipEncoder(name)
    if name in {"mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b"}:
        from .mobileclip_encoder import MobileClipEncoder

        return MobileClipEncoder(name)
    if name in {"siglip_base", "siglip_so400m"}:
        from .siglip_encoder import SigLipEncoder

        return SigLipEncoder(name)
    raise ValueError(f"Unknown encoder backend: {name}")
