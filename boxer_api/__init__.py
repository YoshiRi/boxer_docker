"""Stable integration-facing API surface for Boxer."""

from .adapters import frame_input_from_datum, iter_frame_inputs_from_source
from .config import BoxerConfig, DetectorConfig, PipelineConfig, TrackingConfig
from .detectors import OwlDetector
from .engine import BoxerInferenceEngine, BoxerInferenceRequest
from .pipeline import BoxerPipeline
from .types import Detection2D, Detection3D, FrameInput, FrameResult, PipelineResult, Track3D

__all__ = [
    "BoxerConfig",
    "BoxerInferenceEngine",
    "BoxerInferenceRequest",
    "BoxerPipeline",
    "DetectorConfig",
    "OwlDetector",
    "PipelineConfig",
    "TrackingConfig",
    "Detection2D",
    "Detection3D",
    "FrameInput",
    "FrameResult",
    "PipelineResult",
    "Track3D",
    "frame_input_from_datum",
    "iter_frame_inputs_from_source",
]
