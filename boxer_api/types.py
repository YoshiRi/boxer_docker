from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FrameInput:
    image_bgr: np.ndarray
    camera: Any
    pose_world_rig: Any
    sparse_points_world: Any
    timestamp_ns: int
    rotated: bool = False
    source_name: str | None = None
    device_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Detection2D:
    xyxy: np.ndarray
    label: str
    score: float
    sem_id: int | None = None
    instance_id: int | None = None


@dataclass(slots=True)
class Detection3D:
    center_xyz: np.ndarray
    quaternion_wxyz: np.ndarray
    size_xyz: np.ndarray
    label: str
    score: float
    sem_id: int | None = None
    instance_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Track3D:
    track_id: int
    detection: Detection3D
    support_count: int | None = None
    missed_count: int | None = None
    accumulated_weight: float | None = None


@dataclass(slots=True)
class FrameResult:
    timestamp_ns: int
    detections_2d: list[Detection2D] = field(default_factory=list)
    detections_3d: list[Detection3D] = field(default_factory=list)
    tracks_3d: list[Track3D] = field(default_factory=list)
    rendered_frame_bgr: np.ndarray | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    sequence_name: str
    frames: list[FrameResult] = field(default_factory=list)
    output_root: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
