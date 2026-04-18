"""Clean sensor-agnostic API for running the Boxer 3D detection pipeline.

Typical usage
-------------
    from boxer_pipeline import BoxerPipeline, SensorFrame, Intrinsics

    pipeline = BoxerPipeline(
        ckpt_path="ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt",
        detector="IDEA-Research/grounding-dino-base",  # HF model ID or "owl"
        labels=["chair", "table", "monitor"],
        track=True,
    )

    @pipeline.on_result
    def handle(result):
        for det in result.detections:
            print(f"{det.label}: center={det.center_xyz}, size={det.size_xyz}")

    with pipeline:
        for rgb, depth, intrinsics in my_sensor.stream():
            pipeline.push(SensorFrame(rgb=rgb, depth=depth, intrinsics=intrinsics))

Input contract
--------------
SensorFrame.rgb   — np.ndarray H×W×3 uint8, **RGB** order
SensorFrame.depth — np.ndarray H×W float32 in **metres** (optional)
SensorFrame.pose  — 4×4 float32 world-from-camera transform (optional)
                    If omitted, camera frame is used as world frame.

Output contract
---------------
BoxerResult.detections — list of Detection3D, one per 3D box
BoxerResult.viz_jpg    — JPEG bytes of the visualisation frame (if viz enabled)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from input_sources.frame_source import (
    FrameSourceInfo,
    PushFrameSource,
    build_frame_datum,
    build_identity_pose,
    build_pinhole_camera,
    pose_from_matrix,
)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class Intrinsics:
    """Pinhole camera intrinsics."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float | None = None  # defaults to (width-1)/2
    cy: float | None = None  # defaults to (height-1)/2


@dataclass
class SensorFrame:
    """One camera frame from any sensor."""
    rgb: np.ndarray                  # H×W×3 uint8, RGB order
    intrinsics: Intrinsics
    timestamp_ns: int = field(default_factory=time.time_ns)
    depth: np.ndarray | None = None  # H×W float32, metres
    pose: np.ndarray | None = None   # 4×4 float32, world-from-camera


@dataclass
class Detection3D:
    """Single 3D oriented bounding box detection."""
    label: str
    confidence: float
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]          # (dx, dy, dz) in metres
    rotation_quat_wxyz: tuple[float, float, float, float]


@dataclass
class BoxerResult:
    """Detections for one frame, emitted via on_result callbacks."""
    timestamp_ns: int
    detections: list[Detection3D]
    viz_jpg: bytes | None = None  # JPEG visualisation frame, if skip_viz=False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_SDP_POINTS = 10_000


def _depth_to_sdp_w(
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    pose: np.ndarray | None,
) -> torch.Tensor:
    """Back-project depth map to world-frame semi-dense points."""
    H, W = depth_m.shape
    cx = intrinsics.cx if intrinsics.cx is not None else (W - 1) * 0.5
    cy = intrinsics.cy if intrinsics.cy is not None else (H - 1) * 0.5

    valid = (depth_m > 0.05) & (depth_m < 15.0) & np.isfinite(depth_m)
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return torch.zeros(0, 3, dtype=torch.float32)

    z = depth_m[ys, xs].astype(np.float32)
    x_cam = (xs - cx) / intrinsics.fx * z
    y_cam = (ys - cy) / intrinsics.fy * z
    pts = np.stack([x_cam, y_cam, z], axis=1)  # (N, 3) in camera frame

    if pose is not None:
        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, 3].astype(np.float32)
        pts = pts @ R.T + t

    if len(pts) > _MAX_SDP_POINTS:
        idx = np.random.choice(len(pts), _MAX_SDP_POINTS, replace=False)
        pts = pts[idx]

    return torch.from_numpy(pts.astype(np.float32))


def _obb_to_detections(obb_pr_w) -> list[Detection3D]:
    """Convert ObbTW batch → list of Detection3D."""
    from utils.tw.pose import rotmat_to_quat

    detections = []
    labels = obb_pr_w.text_string()
    if isinstance(labels, str):
        labels = [labels]

    centers = obb_pr_w.bb3_center_world  # (N, 3)
    diagonals = obb_pr_w.bb3_diagonal    # (N, 3)
    probs = obb_pr_w.prob.squeeze(-1)    # (N,)

    for i in range(len(obb_pr_w)):
        R_mat = obb_pr_w[i:i+1].T_world_object.R[0].cpu().numpy()
        qwxyz = rotmat_to_quat(R_mat)
        c = centers[i].cpu().tolist()
        s = diagonals[i].cpu().tolist()
        detections.append(Detection3D(
            label=labels[i],
            confidence=float(probs[i].item()),
            center_xyz=(c[0], c[1], c[2]),
            size_xyz=(s[0], s[1], s[2]),
            rotation_quat_wxyz=qwxyz,
        ))
    return detections


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class BoxerPipeline:
    """Thread-safe pipeline: push SensorFrames in, get BoxerResults out.

    Args:
        ckpt_path:   Path to BoxerNet checkpoint.
        detector:    "owl" to use the built-in OWLv2, or a HuggingFace model ID
                     (e.g. "IDEA-Research/grounding-dino-base",
                           "google/owlv2-base-patch16-ensemble").
        labels:      Text labels / taxonomy name (e.g. ["chair","table"] or ["lvisplus"]).
        thresh2d:    2D confidence threshold.
        thresh3d:    3D confidence threshold.
        device:      "cuda" / "cpu" / "mps" — auto-detected if None.
        track:       Enable online 3D box tracking.
        skip_viz:    Disable frame visualisation (faster).
        output_dir:  Directory for CSV and viz outputs.
        stream_name: Name prefix for output files.
        queue_size:  Internal frame queue capacity.
    """

    def __init__(
        self,
        ckpt_path: str = "ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt",
        *,
        detector: str = "owl",
        labels: list[str] | None = None,
        thresh2d: float = 0.25,
        thresh3d: float = 0.5,
        device: str | None = None,
        track: bool = False,
        skip_viz: bool = False,
        output_dir: str = "output/",
        stream_name: str = "sensor",
        queue_size: int = 8,
    ):
        self.ckpt_path = ckpt_path
        self.detector = detector
        self.labels = labels or ["lvisplus"]
        self.thresh2d = thresh2d
        self.thresh3d = thresh3d
        self.device = device
        self.track = track
        self.skip_viz = skip_viz
        self.output_dir = output_dir
        self.stream_name = stream_name
        self.queue_size = queue_size

        self._callbacks: list[Callable[[BoxerResult], None]] = []
        self._source: PushFrameSource | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def push(self, frame: SensorFrame) -> None:
        """Push one frame into the pipeline (thread-safe, non-blocking)."""
        if self._source is None:
            raise RuntimeError("Call start() or use BoxerPipeline as a context manager first.")
        cam = build_pinhole_camera(
            frame.intrinsics.width,
            frame.intrinsics.height,
            fx=frame.intrinsics.fx,
            fy=frame.intrinsics.fy,
            cx=frame.intrinsics.cx,
            cy=frame.intrinsics.cy,
        )
        pose = pose_from_matrix(frame.pose) if frame.pose is not None else build_identity_pose()
        sdp_w = (
            _depth_to_sdp_w(frame.depth, frame.intrinsics, frame.pose)
            if frame.depth is not None
            else torch.zeros(0, 3, dtype=torch.float32)
        )
        # Convert RGB → BGR for build_frame_datum
        img_bgr = frame.rgb[:, :, ::-1].copy()
        datum = build_frame_datum(
            img_bgr=img_bgr,
            timestamp_ns=frame.timestamp_ns,
            camera=cam,
            pose=pose,
            sdp_w=sdp_w,
        )
        self._source.push_datum(datum)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def on_result(self, callback: Callable[[BoxerResult], None]) -> Callable:
        """Register a callback for 3D detections (decorator or direct call).

        The callback receives a BoxerResult and is called from the
        pipeline background thread.
        """
        self._callbacks.append(callback)
        return callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "BoxerPipeline":
        """Start the background inference thread."""
        if self._started:
            return self
        info = FrameSourceInfo(kind="push", sequence_name=self.stream_name, is_live=True)
        self._source = PushFrameSource(info, queue_size=self.queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True, name="boxer-pipeline")
        self._thread.start()
        self._started = True
        return self

    def stop(self) -> None:
        """Stop the pipeline and wait for the background thread to finish."""
        if not self._started:
            return
        if self._source is not None:
            self._source.close()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        self._started = False

    def __enter__(self) -> "BoxerPipeline":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread: runs run_boxer.run_with_args() with injected source."""
        import argparse
        from run_boxer import run_with_args

        # Build a minimal args namespace compatible with run_with_args
        args = argparse.Namespace(
            input=self.stream_name,
            input_mode="push",
            stream_name=self.stream_name,
            skip_n=1,
            start_n=1,
            max_n=99999,
            pinhole=False,
            camera="rgb",
            detector="owl" if self.detector == "owl" else "hf",
            hf_model=self.detector if self.detector != "owl" else "IDEA-Research/grounding-dino-base",
            thresh2d=self.thresh2d,
            thresh3d=self.thresh3d,
            labels=self.labels,
            detector_hw=960,
            write_name=self.stream_name,
            skip_viz=self.skip_viz,
            cache2d=False,
            cache3d=False,
            no_sdp=False,
            no_csv=False,
            force_cpu=(self.device == "cpu"),
            gt2d=False,
            fuse=False,
            track=self.track,
            ckpt=self.ckpt_path,
            force_precision=None,
            output_dir=self.output_dir,
        )

        def _output_callback(obb_pr_w, time_ns: int, viz_jpg: bytes | None = None) -> None:
            if not self._callbacks:
                return
            try:
                detections = _obb_to_detections(obb_pr_w)
            except Exception:
                detections = []
            result = BoxerResult(
                timestamp_ns=time_ns,
                detections=detections,
                viz_jpg=viz_jpg,
            )
            for cb in self._callbacks:
                try:
                    cb(result)
                except Exception as exc:
                    print(f"[BoxerPipeline] callback error: {exc}")

        try:
            run_with_args(args, source=self._source, output_callback=_output_callback)
        except Exception as exc:
            print(f"[BoxerPipeline] inference thread error: {exc}")
            raise
