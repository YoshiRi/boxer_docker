from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from input_sources import infer_sequence_name, resolve_input_source
from utils.image import torch2cv2

from .types import FrameInput


def frame_input_from_datum(datum: dict[str, Any], *, source: Any | None = None) -> FrameInput:
    """Convert an existing runner datum into the public API frame type."""

    rotated_value = datum.get("rotated0", False)
    rotated = bool(rotated_value.item()) if hasattr(rotated_value, "item") else bool(rotated_value)

    return FrameInput(
        image_bgr=torch2cv2(datum["img0"], rotate=datum["rotated0"], ensure_rgb=False),
        camera=datum["cam0"],
        pose_world_rig=datum["T_world_rig0"],
        sparse_points_world=datum["sdp_w"],
        timestamp_ns=int(datum["time_ns0"]),
        rotated=rotated,
        source_name=getattr(source, "camera", None),
        device_name=getattr(source, "device_name", None),
        metadata={},
    )


def iter_frame_inputs_from_source(source: Iterable[dict[str, Any]]) -> Iterator[FrameInput]:
    """Yield public API frame objects from an existing Boxer source iterable."""

    for datum in source:
        if datum is False:
            continue
        yield frame_input_from_datum(datum, source=source)


def resolve_frame_inputs(args) -> tuple[str, Iterator[FrameInput]]:
    """Resolve existing CLI args into an API-friendly sequence name and frame iterator."""

    resolved = resolve_input_source(args)
    sequence_name = infer_sequence_name(args)
    return sequence_name or resolved.sequence_name, iter_frame_inputs_from_source(
        resolved.source
    )
