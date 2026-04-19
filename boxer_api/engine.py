from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from boxernet.boxernet import BoxerNet
from input_sources.frame_source import build_frame_datum
from utils.demo_utils import CKPT_PATH

from .config import BoxerConfig, DetectorConfig
from .detectors import OwlDetector, make_detector
from .types import Detection2D, Detection3D, FrameInput, FrameResult


@dataclass(slots=True)
class BoxerInferenceRequest:
    frame: FrameInput
    detections_2d: list[Detection2D] | None = None
    detector: DetectorConfig | None = None
    boxer: BoxerConfig | None = None


class BoxerInferenceEngine:
    """Single-frame Boxer inference boundary.

    The implementation will be extracted incrementally from run_boxer.py.
    """

    def __init__(
        self,
        *,
        detector: DetectorConfig | None = None,
        boxer: BoxerConfig | None = None,
    ) -> None:
        self.detector_config = detector or DetectorConfig()
        self.boxer_config = boxer or BoxerConfig()
        self._device = self._select_device(self.boxer_config)
        self._boxernet: BoxerNet | None = None
        self._detector = make_detector(self.detector_config.detector_name, device=self._device)
        self._loaded_detector_name = self.detector_config.detector_name

    def infer_frame(self, request: BoxerInferenceRequest) -> FrameResult:
        detector_cfg = request.detector or self.detector_config
        boxer_cfg = request.boxer or self.boxer_config
        frame = request.frame

        timings_ms: dict[str, float] = {}
        detections_2d = list(request.detections_2d) if request.detections_2d is not None else None
        if detections_2d is None:
            detections_2d, detect_ms = self._detect_2d(frame, detector_cfg)
            timings_ms["owl"] = round(detect_ms, 3)
        if not detections_2d:
            return FrameResult(
                timestamp_ns=frame.timestamp_ns,
                detections_2d=[],
                detections_3d=[],
                timings_ms=timings_ms,
                metadata={"device": self._device, "detector_name": detector_cfg.detector_name},
            )

        boxernet = self._ensure_boxernet(boxer_cfg)
        datum = build_frame_datum(
            img_bgr=frame.image_bgr,
            timestamp_ns=frame.timestamp_ns,
            camera=frame.camera,
            pose=frame.pose_world_rig,
            resize=boxernet.hw,
            rotated=frame.rotated,
            sdp_w=None if boxer_cfg.disable_sparse_depth else frame.sparse_points_world,
        )

        bb2d, scores2d, labels2d, sem_ids2d = self._prepare_2d_inputs(detections_2d)
        datum["bb2d"] = bb2d

        t0 = time.perf_counter()
        if boxer_cfg.force_precision is not None:
            precision_dtype = (
                torch.bfloat16
                if boxer_cfg.force_precision == "bfloat16"
                else torch.float32
            )
        elif self._device == "cuda" and torch.cuda.is_bf16_supported():
            precision_dtype = torch.bfloat16
        else:
            precision_dtype = torch.float32

        if self._device in {"mps", "cpu"}:
            outputs = boxernet.forward(datum)
        else:
            with torch.autocast(device_type=self._device, dtype=precision_dtype):
                outputs = boxernet.forward(datum)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        obb_pr_w = outputs["obbs_pr_w"].cpu()[0]
        sem_ids = torch.tensor(sem_ids2d, dtype=torch.int32)
        obb_pr_w.set_sem_id(sem_ids)

        scores3d = obb_pr_w.prob.squeeze(-1).clone()
        keepers = obb_pr_w.prob.squeeze(-1) >= boxer_cfg.threshold_3d
        obb_pr_w = obb_pr_w[keepers].clone()
        scores3d = scores3d[keepers].clone()
        labels3d = [labels2d[i] for i in range(len(labels2d)) if keepers[i]]
        kept_scores2d = scores2d[keepers]
        mean_scores = (kept_scores2d + scores3d) / 2.0
        obb_pr_w.set_prob(mean_scores)

        detections_3d = [
            self._obb_to_detection3d(obb_pr_w[i], labels3d[i])
            for i in range(len(labels3d))
        ]

        return FrameResult(
            timestamp_ns=frame.timestamp_ns,
            detections_2d=detections_2d,
            detections_3d=detections_3d,
            timings_ms={**timings_ms, "boxer": round(inference_ms, 3)},
            metadata={
                "device": self._device,
                "detector_name": detector_cfg.detector_name,
                "source_name": frame.source_name,
                "device_name": frame.device_name,
                "image_width": int(frame.image_bgr.shape[1]),
                "image_height": int(frame.image_bgr.shape[0]),
            },
        )

    @staticmethod
    def _select_device(boxer_cfg: BoxerConfig) -> str:
        if torch.backends.mps.is_available() and not boxer_cfg.force_cpu:
            return "mps"
        if torch.cuda.is_available() and not boxer_cfg.force_cpu:
            return "cuda"
        return "cpu"

    def _ensure_boxernet(self, boxer_cfg: BoxerConfig) -> BoxerNet:
        checkpoint_path = boxer_cfg.checkpoint_path or (
            f"{CKPT_PATH}/boxernet_hw960in4x6d768-wssxpf9p.ckpt"
        )
        if self._boxernet is None:
            self._boxernet = BoxerNet.load_from_checkpoint(
                checkpoint_path,
                device=self._device,
            )
        return self._boxernet

    def _detect_2d(
        self,
        frame: FrameInput,
        detector_cfg: DetectorConfig,
    ) -> tuple[list[Detection2D], float]:
        # Re-create the detector only when the requested backend name changes.
        if self._loaded_detector_name != detector_cfg.detector_name:
            self._detector = make_detector(detector_cfg.detector_name, device=self._device)
            self._loaded_detector_name = detector_cfg.detector_name
        return self._detector.detect(frame, detector_cfg)

    @staticmethod
    def _prepare_2d_inputs(
        detections_2d: list[Detection2D],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
        label_to_sem_id: dict[str, int] = {}
        boxes = []
        scores = []
        labels = []
        sem_ids = []
        for detection in detections_2d:
            xyxy = np.asarray(detection.xyxy, dtype=np.float32).reshape(4)
            boxes.append([xyxy[0], xyxy[2], xyxy[1], xyxy[3]])
            scores.append(float(detection.score))
            labels.append(detection.label)
            if detection.sem_id is not None:
                sem_ids.append(int(detection.sem_id))
                continue
            if detection.label not in label_to_sem_id:
                label_to_sem_id[detection.label] = len(label_to_sem_id)
            sem_ids.append(label_to_sem_id[detection.label])
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(scores, dtype=torch.float32),
            labels,
            sem_ids,
        )

    @staticmethod
    def _obb_to_detection3d(obb, label: str) -> Detection3D:
        from utils.tw.pose import rotmat_to_quat

        center = obb.bb3_center_world.detach().cpu().numpy().reshape(3)
        R = obb.T_world_object.R.detach().cpu().numpy().reshape(3, 3)
        qwxyz = rotmat_to_quat(R)
        size = obb.bb3_diagonal.detach().cpu().numpy().reshape(3)
        return Detection3D(
            center_xyz=center,
            quaternion_wxyz=np.array(qwxyz, dtype=np.float32),
            size_xyz=size,
            label=label,
            score=float(obb.prob.item()),
            sem_id=int(obb.sem_id.item()),
            instance_id=int(obb.inst_id.item()),
            metadata={},
        )
