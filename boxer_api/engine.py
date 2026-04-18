from __future__ import annotations

from dataclasses import dataclass

from .config import BoxerConfig, DetectorConfig
from .types import Detection2D, FrameInput, FrameResult


@dataclass(slots=True)
class BoxerInferenceRequest:
    frame: FrameInput
    detections_2d: list[Detection2D] | None = None
    detector: DetectorConfig | None = None
    boxer: BoxerConfig | None = None


class BoxerInferenceEngine:
    """Single-frame Boxer inference boundary.

    The implementation will be extracted incrementally from run_boxer.py.
    """

    def infer_frame(self, request: BoxerInferenceRequest) -> FrameResult:
        raise NotImplementedError("BoxerInferenceEngine.infer_frame is not implemented yet")
