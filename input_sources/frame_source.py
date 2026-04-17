from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Iterator

import cv2
import numpy as np
import torch

from utils.tw.camera import CameraTW
from utils.tw.pose import PoseTW


BoxerDatum = dict[str, Any]


@dataclass(slots=True)
class FrameSourceInfo:
    kind: str
    sequence_name: str
    camera: str = "rgb"
    device_name: str = "unknown"
    is_live: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseFrameSource(ABC):
    def __init__(self, info: FrameSourceInfo, *, resize: tuple[int, int] | None = None):
        self.info = info
        self.resize = normalize_resize(resize)

    @property
    def camera(self) -> str:
        return self.info.camera

    @property
    def device_name(self) -> str:
        return self.info.device_name

    @property
    def sequence_name(self) -> str:
        return self.info.sequence_name

    def __iter__(self) -> Iterator[BoxerDatum]:
        return self

    def __len__(self) -> int:
        raise TypeError(f"{self.__class__.__name__} does not expose a finite length")

    def set_resize(self, resize: int | tuple[int, int] | None) -> None:
        self.resize = normalize_resize(resize)

    def reset(self) -> None:
        return

    def close(self) -> None:
        return

    @abstractmethod
    def __next__(self) -> BoxerDatum:
        raise StopIteration


class PushFrameSource(BaseFrameSource):
    def __init__(
        self,
        info: FrameSourceInfo,
        *,
        resize: tuple[int, int] | None = None,
        max_frames: int | None = None,
        timeout_sec: float = 1.0,
        queue_size: int = 8,
    ):
        super().__init__(info, resize=resize)
        self.max_frames = max_frames
        self.timeout_sec = timeout_sec
        self._queue: Queue[BoxerDatum | None] = Queue(maxsize=max(1, queue_size))
        self._closed = False
        self._delivered = 0

    def __len__(self) -> int:
        if self.max_frames is None:
            return super().__len__()
        return self.max_frames

    def push_datum(self, datum: BoxerDatum) -> None:
        if self._closed:
            return
        self._queue.put(datum)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)

    def __next__(self) -> BoxerDatum:
        if self.max_frames is not None and self._delivered >= self.max_frames:
            raise StopIteration
        try:
            datum = self._queue.get(timeout=self.timeout_sec)
        except Empty as exc:
            raise StopIteration from exc
        if datum is None:
            raise StopIteration
        self._delivered += 1
        return datum


def source_length(source: Any) -> int | None:
    try:
        return len(source)
    except (TypeError, AttributeError):
        return None


def sanitize_sequence_name(value: str, default: str = "input_source") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or default


def normalize_resize(resize: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if resize is None:
        return None
    if isinstance(resize, int):
        return (resize, resize)
    if len(resize) != 2:
        raise ValueError(f"Resize must be an int or (width, height) tuple, got: {resize}")
    return tuple(int(value) for value in resize)


def image_bgr_to_tensor(img_bgr: np.ndarray, *, resize: int | tuple[int, int] | None = None) -> torch.Tensor:
    resize = normalize_resize(resize)
    if resize is not None:
        img_bgr = cv2.resize(img_bgr, resize, interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).float()[None] / 255.0


def build_identity_pose() -> PoseTW:
    R = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    t = torch.zeros(1, 3, dtype=torch.float32)
    return PoseTW.from_Rt(R, t)


def build_pinhole_camera(
    width: int,
    height: int,
    *,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> CameraTW:
    fx = float(max(width, height)) if fx is None else float(fx)
    fy = float(max(width, height)) if fy is None else float(fy)
    cx = float((width - 1) * 0.5) if cx is None else float(cx)
    cy = float((height - 1) * 0.5) if cy is None else float(cy)
    return CameraTW.from_surreal(
        width=torch.tensor([float(width)], dtype=torch.float32),
        height=torch.tensor([float(height)], dtype=torch.float32),
        type_str="Pinhole",
        params=torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32),
    )


def pose_from_matrix(matrix: Any) -> PoseTW:
    pose = torch.tensor(matrix, dtype=torch.float32)
    if pose.shape == (4, 4):
        pose = pose[:3, :]
    if pose.shape == (3, 4):
        return PoseTW.from_matrix3x4(pose.unsqueeze(0))
    if pose.numel() == 12:
        return PoseTW(pose.reshape(12))
    if pose.numel() == 16:
        return PoseTW.from_matrix3x4(pose.reshape(4, 4)[:3, :].unsqueeze(0))
    raise ValueError(f"Unsupported pose matrix shape: {tuple(pose.shape)}")


def build_frame_datum(
    *,
    img_bgr: np.ndarray,
    timestamp_ns: int,
    camera: CameraTW,
    pose: PoseTW | None = None,
    resize: tuple[int, int] | None = None,
    rotated: bool = False,
    sdp_w: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
) -> BoxerDatum:
    resize = normalize_resize(resize)
    frame_camera = camera.scale_to_size(resize) if resize is not None else camera
    datum: BoxerDatum = {
        "img0": image_bgr_to_tensor(img_bgr, resize=resize),
        "cam0": frame_camera,
        "T_world_rig0": pose if pose is not None else build_identity_pose(),
        "sdp_w": sdp_w if sdp_w is not None else torch.zeros(0, 3, dtype=torch.float32),
        "time_ns0": int(timestamp_ns),
        "rotated0": torch.tensor(bool(rotated)).reshape(1),
    }
    if extra:
        datum.update(extra)
    return datum
