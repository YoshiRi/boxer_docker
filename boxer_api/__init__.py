"""Stable integration-facing API surface for Boxer."""

from .adapters import frame_input_from_datum
from .config import BoxerConfig, DetectorConfig, PipelineConfig, TrackingConfig
from .engine import BoxerInferenceEngine, BoxerInferenceRequest
from .types import Detection2D, Detection3D, FrameInput, FrameResult, PipelineResult, Track3D

__all__ = [
    "BoxerConfig",
    "BoxerInferenceEngine",
    "BoxerInferenceRequest",
    "DetectorConfig",
    "PipelineConfig",
    "TrackingConfig",
    "Detection2D",
    "Detection3D",
    "FrameInput",
    "FrameResult",
    "PipelineResult",
    "Track3D",
    "frame_input_from_datum",
]
