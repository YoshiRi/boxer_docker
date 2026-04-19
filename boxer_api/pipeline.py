from __future__ import annotations

from collections.abc import Callable, Iterable

from .config import PipelineConfig
from .engine import BoxerInferenceEngine, BoxerInferenceRequest
from .types import Detection2D, FrameInput, PipelineResult

DetectionProvider = Callable[[FrameInput], Iterable[Detection2D]]

class BoxerPipeline:
    """Sequence-level Boxer orchestration boundary.

    The implementation will be extracted incrementally from run_boxer.py and
    scripts/run_boxer_job.py.
    """

    def __init__(
        self,
        *,
        engine: BoxerInferenceEngine | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.engine = engine or BoxerInferenceEngine()
        self.config = config or PipelineConfig()

    def run_sequence(
        self,
        frames: Iterable[FrameInput],
        *,
        sequence_name: str,
        detections_provider: DetectionProvider | None = None,
        config: PipelineConfig | None = None,
    ) -> PipelineResult:
        pipeline_config = config or self.config
        result = PipelineResult(sequence_name=sequence_name, metadata={})

        for frame in frames:
            detections_2d = None
            if detections_provider is not None:
                detections_2d = list(detections_provider(frame))
            frame_result = self.engine.infer_frame(
                BoxerInferenceRequest(
                    frame=frame,
                    detections_2d=detections_2d,
                    detector=pipeline_config.detector,
                    boxer=pipeline_config.boxer,
                )
            )
            result.frames.append(frame_result)

        result.metadata["num_frames"] = len(result.frames)
        result.metadata["write_name"] = pipeline_config.write_name
        return result
