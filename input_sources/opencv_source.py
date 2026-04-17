from __future__ import annotations

from pathlib import Path

import cv2

from .frame_source import (
    BaseFrameSource,
    FrameSourceInfo,
    build_frame_datum,
    build_identity_pose,
    build_pinhole_camera,
    sanitize_sequence_name,
)


class OpenCvCaptureFrameSource(BaseFrameSource):
    def __init__(
        self,
        input_path: str,
        *,
        start_frame: int = 0,
        skip_frames: int = 1,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
        camera_name: str = "rgb",
        width: int | None = None,
        height: int | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        frame_period_ns: int = 100_000_000,
        start_time_ns: int = 0,
        sequence_name: str | None = None,
    ):
        self.input_path = input_path
        self.start_frame = max(0, start_frame)
        self.skip_frames = max(1, skip_frames)
        self.max_frames = max_frames
        self.frame_period_ns = int(frame_period_ns)
        self.start_time_ns = int(start_time_ns)
        self._delivered = 0
        self._capture = None
        self._camera = None
        self._width = width
        self._height = height
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._sequence_name = sequence_name or sanitize_sequence_name(str(input_path))
        super().__init__(
            FrameSourceInfo(
                kind="cv2",
                sequence_name=self._sequence_name,
                camera=camera_name,
                device_name="opencv-capture",
                is_live=not Path(input_path).exists(),
            ),
            resize=resize,
        )
        self.reset()

    def __len__(self) -> int:
        if self.max_frames is not None:
            return self.max_frames
        if self._capture is None:
            return super().__len__()
        total = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return super().__len__()
        remaining = max(0, total - self.start_frame)
        return (remaining + self.skip_frames - 1) // self.skip_frames

    def _capture_target(self):
        if self.input_path.isdigit() and not Path(self.input_path).exists():
            return int(self.input_path)
        return self.input_path

    def _open_capture(self):
        capture = cv2.VideoCapture(self._capture_target())
        if not capture.isOpened():
            raise ValueError(f"Failed to open OpenCV capture source: {self.input_path}")
        for _ in range(self.start_frame):
            ok, _ = capture.read()
            if not ok:
                break
        return capture

    def _ensure_camera(self, frame):
        if self._camera is not None:
            return
        frame_h, frame_w = frame.shape[:2]
        self._camera = build_pinhole_camera(
            self._width or frame_w,
            self._height or frame_h,
            fx=self._fx,
            fy=self._fy,
            cx=self._cx,
            cy=self._cy,
        )

    def reset(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = self._open_capture()
        self._camera = None
        self._delivered = 0

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __next__(self):
        if self.max_frames is not None and self._delivered >= self.max_frames:
            raise StopIteration
        if self._capture is None:
            raise StopIteration
        frame = None
        for _ in range(self.skip_frames):
            ok, frame = self._capture.read()
            if not ok:
                self.close()
                raise StopIteration
        assert frame is not None
        self._ensure_camera(frame)
        timestamp_ns = self.start_time_ns + self._delivered * self.frame_period_ns
        datum = build_frame_datum(
            img_bgr=frame,
            timestamp_ns=timestamp_ns,
            camera=self._camera,
            pose=build_identity_pose(),
            resize=self.resize,
        )
        self._delivered += 1
        return datum
