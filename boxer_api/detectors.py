from __future__ import annotations

import time

import numpy as np

from input_sources.frame_source import build_frame_datum, image_bgr_to_tensor
from owl.owl_wrapper import OwlWrapper
from utils.taxonomy import load_text_labels

from .config import DetectorConfig
from .types import Detection2D, FrameInput


def _boxes_to_detections(
    bb2d,
    scores2d,
    label_ints,
    text_labels: list[str],
) -> list[Detection2D]:
    """Convert raw detector outputs to Detection2D list.

    Handles the (x1,x2,y1,y2) → (x1,y1,x2,y2) box convention flip.
    """
    detections: list[Detection2D] = []
    for idx in range(len(label_ints)):
        box = bb2d[idx]
        detections.append(
            Detection2D(
                xyxy=np.array([box[0], box[2], box[1], box[3]], dtype=np.float32),
                label=text_labels[int(label_ints[idx])],
                score=float(scores2d[idx]),
                sem_id=int(label_ints[idx]),
            )
        )
    return detections


class OwlDetector:
    """API-facing adapter around the repository's built-in OWLv2 wrapper."""

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
        text_labels = load_text_labels(config.labels)
        image_torch = image_bgr_to_tensor(frame.image_bgr)

        t0 = time.perf_counter()
        bb2d, scores2d, label_ints, _ = owl.forward(
            image_torch * 255.0,
            frame.rotated,
            resize_to_HW=(config.detector_hw, config.detector_hw),
        )
        detect_ms = (time.perf_counter() - t0) * 1000.0

        return _boxes_to_detections(bb2d, scores2d, label_ints, text_labels), detect_ms

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


class HFDetector:
    """API-facing adapter around HuggingFace zero-shot detectors.

    Supports Grounding DINO (``IDEA-Research/grounding-dino-base``) and
    OWLv2 via HuggingFace (``google/owlv2-base-patch16-ensemble``).

    Requires ``transformers`` and ``timm``:
        pip install transformers timm
    """

    def __init__(self, *, device: str) -> None:
        self.device = device
        self._model_id: str | None = None
        self._detector = None
        self._signature: tuple[tuple[str, ...], float] | None = None

    def detect(
        self,
        frame: FrameInput,
        config: DetectorConfig,
    ) -> tuple[list[Detection2D], float]:
        from detectors.hf_detector import HFDetector as _HFDet

        text_labels = load_text_labels(config.labels)
        model_id = config.hf_model_id
        signature = (tuple(text_labels), float(config.threshold_2d))

        if self._detector is None or self._model_id != model_id or self._signature != signature:
            self._detector = _HFDet(
                model_id=model_id,
                device=self.device,
                text_prompts=text_labels,
                min_confidence=config.threshold_2d,
            )
            self._model_id = model_id
            self._signature = signature

        image_torch = image_bgr_to_tensor(frame.image_bgr)

        t0 = time.perf_counter()
        bb2d, scores2d, label_ints, _ = self._detector.forward(
            image_torch * 255.0,
            frame.rotated,
        )
        detect_ms = (time.perf_counter() - t0) * 1000.0

        return _boxes_to_detections(bb2d, scores2d, label_ints, text_labels), detect_ms


def make_detector(detector_name: str, *, device: str) -> OwlDetector | HFDetector:
    """Factory: instantiate the right detector from DetectorConfig.detector_name."""
    if detector_name == "owl":
        return OwlDetector(device=device)
    return HFDetector(device=device)
