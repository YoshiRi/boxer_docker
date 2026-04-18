from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import PipelineConfig
from .types import FrameInput, PipelineResult

class BoxerPipeline:
    """Sequence-level Boxer orchestration boundary.

    The implementation will be extracted incrementally from run_boxer.py and
    scripts/run_boxer_job.py.
    """


    def run_sequence(
        self,
        frames: Iterable[FrameInput],
        *,
        sequence_name: str,
        config: PipelineConfig | None = None,
    ) -> PipelineResult:
        raise NotImplementedError("BoxerPipeline.run_sequence is not implemented yet")
