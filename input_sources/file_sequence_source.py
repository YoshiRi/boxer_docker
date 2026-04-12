from __future__ import annotations

import json
from pathlib import Path

import cv2

from .frame_source import (
    BaseFrameSource,
    FrameSourceInfo,
    build_frame_datum,
    build_identity_pose,
    build_pinhole_camera,
    pose_from_matrix,
    sanitize_sequence_name,
)


class ImageSequenceFrameSource(BaseFrameSource):
    def __init__(
        self,
        input_path: str,
        *,
        image_glob: str = "*.jpg,*.jpeg,*.png",
        metadata_path: str | None = None,
        start_frame: int = 0,
        skip_frames: int = 1,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
        camera_name: str = "rgb",
        device_name: str = "file-replay",
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
        self.root = Path(input_path).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Image sequence input does not exist: {self.root}")

        self.frame_period_ns = int(frame_period_ns)
        self.start_time_ns = int(start_time_ns)
        self._frame_specs = self._load_frame_specs(
            metadata_path=metadata_path,
            image_glob=image_glob,
        )
        self._frame_specs = self._frame_specs[start_frame:: max(1, skip_frames)]
        if max_frames is not None:
            self._frame_specs = self._frame_specs[:max_frames]
        if not self._frame_specs:
            raise ValueError(f"No image frames found under {self.root}")

        self._index = 0
        first_img = cv2.imread(str(self._frame_specs[0]["path"]), cv2.IMREAD_COLOR)
        if first_img is None:
            raise ValueError(f"Failed to load first image: {self._frame_specs[0]['path']}")
        inferred_h, inferred_w = first_img.shape[:2]
        self._camera_model = build_pinhole_camera(
            width or inferred_w,
            height or inferred_h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        super().__init__(
            FrameSourceInfo(
                kind="file",
                sequence_name=sequence_name or sanitize_sequence_name(self.root.name),
                camera=camera_name,
                device_name=device_name,
                is_live=False,
            ),
            resize=resize,
        )

    def __len__(self) -> int:
        return len(self._frame_specs)

    def reset(self) -> None:
        self._index = 0

    def _load_frame_specs(self, *, metadata_path: str | None, image_glob: str):
        if metadata_path is not None:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            frame_specs = []
            for idx, frame in enumerate(metadata.get("frames", [])):
                rel_path = frame.get("path") or frame.get("file")
                if not rel_path:
                    raise ValueError("Frame metadata entries must provide 'path' or 'file'")
                frame_specs.append(
                    {
                        "path": (self.root / rel_path).resolve(),
                        "timestamp_ns": frame.get(
                            "timestamp_ns",
                            self.start_time_ns + idx * self.frame_period_ns,
                        ),
                        "pose": frame.get("T_world_rig0") or frame.get("pose_matrix"),
                    }
                )
            return frame_specs

        if self.root.is_file():
            return [
                {
                    "path": self.root,
                    "timestamp_ns": self.start_time_ns,
                    "pose": None,
                }
            ]

        patterns = [p.strip() for p in image_glob.split(",") if p.strip()]
        frame_paths = []
        for pattern in patterns:
            frame_paths.extend(sorted(self.root.glob(pattern)))
        return [
            {
                "path": path.resolve(),
                "timestamp_ns": self.start_time_ns + idx * self.frame_period_ns,
                "pose": None,
            }
            for idx, path in enumerate(frame_paths)
        ]

    def __next__(self):
        if self._index >= len(self._frame_specs):
            raise StopIteration
        spec = self._frame_specs[self._index]
        img = cv2.imread(str(spec["path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image frame: {spec['path']}")
        pose = pose_from_matrix(spec["pose"]) if spec["pose"] is not None else build_identity_pose()
        datum = build_frame_datum(
            img_bgr=img,
            timestamp_ns=int(spec["timestamp_ns"]),
            camera=self._camera_model,
            pose=pose,
            resize=self.resize,
        )
        self._index += 1
        return datum
