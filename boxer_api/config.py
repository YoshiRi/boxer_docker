from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DetectorConfig:
    detector_name: str = "owl"
    labels: list[str] = field(default_factory=lambda: ["lvisplus"])
    threshold_2d: float = 0.25
    detector_hw: int = 960
    force_precision: str | None = None
    hf_model_id: str = "IDEA-Research/grounding-dino-base"


@dataclass(slots=True)
class BoxerConfig:
    threshold_3d: float = 0.5
    checkpoint_path: str | None = None
    force_cpu: bool = False
    force_precision: str | None = None
    disable_sparse_depth: bool = False


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool = False
    iou_threshold: float = 0.25
    min_hits: int = 8
    confidence_threshold: float = 0.5
    samples_per_dim: int = 8
    max_missed: int = 90


@dataclass(slots=True)
class PipelineConfig:
    write_name: str = "boxer"
    skip_visualization: bool = False
    write_csv: bool = True
    enable_fusion: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    boxer: BoxerConfig = field(default_factory=BoxerConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
