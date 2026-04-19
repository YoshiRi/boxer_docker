from __future__ import annotations

import time

import numpy as np

from input_sources.frame_source import build_frame_datum
from owl.owl_wrapper import OwlWrapper
from utils.taxonomy import load_text_labels

from .config import DetectorConfig
from .types import Detection2D, FrameInput


class OwlDetector:
    """API-facing adapter around the repository's OWLv2 wrapper."""

    def __init__(self, *, device: str) -> None:
        self.device = device
        self._owl: OwlWrapper | None = None
        self._signature: tuple[tuple[str, ...], float, str | None] | None = None

    def detect(
        self,
        frame: FrameInput,
        config: DetectorConfig,
    ) -> tuple[list[Detection2D], float]:
        owl = self._ensure_owl(config)
        image_torch = build_frame_datum(
            img_bgr=frame.image_bgr,
            timestamp_ns=frame.timestamp_ns,
            camera=frame.camera,
            pose=frame.pose_world_rig,
            resize=None,
            rotated=frame.rotated,
            sdp_w=frame.sparse_points_world,
        )["img0"]
        text_labels = load_text_labels(config.labels)

        t0 = time.perf_counter()
        bb2d, scores2d, label_ints, _ = owl.forward(
            image_torch * 255.0,
            frame.rotated,
            resize_to_HW=(config.detector_hw, config.detector_hw),
        )
        detect_ms = (time.perf_counter() - t0) * 1000.0

        detections_2d = []
        for idx in range(len(label_ints)):
            box = bb2d[idx]
            detections_2d.append(
                Detection2D(
                    xyxy=np.array([box[0], box[2], box[1], box[3]], dtype=np.float32),
                    label=text_labels[int(label_ints[idx])],
                    score=float(scores2d[idx]),
                    sem_id=int(label_ints[idx]),
                )
            )
        return detections_2d, detect_ms

    def _ensure_owl(self, config: DetectorConfig) -> OwlWrapper:
        text_labels = load_text_labels(config.labels)
        signature = (
            tuple(text_labels),
            float(config.threshold_2d),
            config.force_precision,
        )
        if self._owl is None or self._signature != signature:
            self._owl = OwlWrapper(
                device=self.device,
                text_prompts=text_labels,
                min_confidence=config.threshold_2d,
                precision=config.force_precision,
            )
            self._signature = signature
        return self._owl
