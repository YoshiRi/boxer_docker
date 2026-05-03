"""Intel RealSense D4xx adapter for BoxerPipeline.

Streams RGB + depth from a RealSense camera and pushes SensorFrames into
a BoxerPipeline.  Depth is used to generate the semi-dense point cloud that
BoxerNet needs for accurate 3D lifting.

Usage:
    from boxer_pipeline import BoxerPipeline
    from input_sources.realsense_source import RealSenseSource

    with BoxerPipeline(...) as pipeline:
        src = RealSenseSource(pipeline, width=640, height=480, fps=30)
        src.run()          # blocks until Ctrl-C or max_frames reached

Requirements:
    pip install pyrealsense2
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from boxer_pipeline import BoxerPipeline, Intrinsics, SensorFrame


class RealSenseSource:
    """Streams an Intel RealSense D4xx camera into a BoxerPipeline.

    Args:
        pipeline:      A started BoxerPipeline instance.
        width:         Stream width in pixels (default 640).
        height:        Stream height in pixels (default 480).
        fps:           Target frame-rate (default 30).
        max_frames:    Stop after this many frames (None = run forever).
        align_depth:   Align depth frames to the colour sensor (recommended).
        serial:        Device serial number — leave None to use first found device.
    """

    def __init__(
        self,
        pipeline: "BoxerPipeline",
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_frames: int | None = None,
        align_depth: bool = True,
        serial: str | None = None,
    ):
        self.boxer_pipeline = pipeline
        self.width = width
        self.height = height
        self.fps = fps
        self.max_frames = max_frames
        self.align_depth = align_depth
        self.serial = serial

    def run(self) -> None:
        """Start streaming; blocks until stopped or max_frames reached."""
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required for RealSenseSource. "
                "Install it with: pip install pyrealsense2"
            ) from exc

        from boxer_pipeline import Intrinsics, SensorFrame

        rs_pipeline = rs.pipeline()
        config = rs.config()

        if self.serial:
            config.enable_device(self.serial)

        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = rs_pipeline.start(config)

        align = rs.align(rs.stream.color) if self.align_depth else None

        # Read intrinsics from the device
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        cam_intrinsics = Intrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
        )

        # Depth scale: converts raw uint16 → metres
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        print(
            f"[RealSenseSource] streaming {self.width}×{self.height}@{self.fps}fps  "
            f"depth_scale={depth_scale:.5f}  "
            f"fx={intr.fx:.1f} fy={intr.fy:.1f}"
        )

        frame_count = 0
        try:
            while self.max_frames is None or frame_count < self.max_frames:
                frames = rs_pipeline.wait_for_frames(timeout_ms=5000)

                if align is not None:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                timestamp_ns = int(color_frame.get_timestamp() * 1_000_000)  # ms → ns
                if timestamp_ns == 0:
                    timestamp_ns = time.time_ns()

                rgb = np.asarray(color_frame.get_data(), dtype=np.uint8)     # H×W×3 RGB
                depth_raw = np.asarray(depth_frame.get_data(), dtype=np.uint16)  # H×W uint16
                depth_m = depth_raw.astype(np.float32) * depth_scale         # H×W metres

                sensor_frame = SensorFrame(
                    rgb=rgb,
                    intrinsics=cam_intrinsics,
                    timestamp_ns=timestamp_ns,
                    depth=depth_m,
                )
                self.boxer_pipeline.push(sensor_frame)
                frame_count += 1

        except KeyboardInterrupt:
            print("[RealSenseSource] interrupted by user")
        finally:
            rs_pipeline.stop()
            self.boxer_pipeline.stop()
            print(f"[RealSenseSource] stopped after {frame_count} frames")
