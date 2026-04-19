from __future__ import annotations

import csv
from pathlib import Path

from .types import PipelineResult


def write_pipeline_csv_artifacts(
    pipeline_result: PipelineResult,
    *,
    output_dir: str,
    write_name: str,
) -> list[dict[str, object]]:
    """Write minimal CSV artifacts for API-backed runs.

    This intentionally covers the machine-readable outputs first and leaves
    visualization/tracking/fusion artifacts for later steps.
    """

    sequence_root = Path(output_dir).expanduser() / pipeline_result.sequence_name
    sequence_root.mkdir(parents=True, exist_ok=True)

    obb_csv = sequence_root / f"{write_name}_3dbbs.csv"
    bb2d_csv = sequence_root / "owl_2dbbs.csv"

    _write_3d_csv(obb_csv, pipeline_result)
    _write_2d_csv(bb2d_csv, pipeline_result)

    artifacts = [
        {
            "name": "boxer_3dbbs_csv",
            "path": str(obb_csv),
            "exists": obb_csv.exists(),
            "size_bytes": obb_csv.stat().st_size if obb_csv.exists() else None,
        },
        {
            "name": "owl_2dbbs_csv",
            "path": str(bb2d_csv),
            "exists": bb2d_csv.exists(),
            "size_bytes": bb2d_csv.stat().st_size if bb2d_csv.exists() else None,
        },
    ]
    pipeline_result.output_root = str(sequence_root)
    pipeline_result.artifacts = artifacts
    return artifacts


def _write_3d_csv(path: Path, pipeline_result: PipelineResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_ns",
                "tx_world_object",
                "ty_world_object",
                "tz_world_object",
                "qw_world_object",
                "qx_world_object",
                "qy_world_object",
                "qz_world_object",
                "scale_x",
                "scale_y",
                "scale_z",
                "name",
                "instance",
                "sem_id",
                "prob",
            ]
        )
        for frame in pipeline_result.frames:
            for detection in frame.detections_3d:
                writer.writerow(
                    [
                        frame.timestamp_ns,
                        f"{float(detection.center_xyz[0]):.6f}",
                        f"{float(detection.center_xyz[1]):.6f}",
                        f"{float(detection.center_xyz[2]):.6f}",
                        f"{float(detection.quaternion_wxyz[0]):.6f}",
                        f"{float(detection.quaternion_wxyz[1]):.6f}",
                        f"{float(detection.quaternion_wxyz[2]):.6f}",
                        f"{float(detection.quaternion_wxyz[3]):.6f}",
                        f"{float(detection.size_xyz[0]):.6f}",
                        f"{float(detection.size_xyz[1]):.6f}",
                        f"{float(detection.size_xyz[2]):.6f}",
                        detection.label,
                        detection.instance_id if detection.instance_id is not None else -1,
                        detection.sem_id if detection.sem_id is not None else -1,
                        f"{float(detection.score):.6f}",
                    ]
                )


def _write_2d_csv(path: Path, pipeline_result: PipelineResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_ns",
                "frame_id",
                "sensor",
                "device",
                "img_width",
                "img_height",
                "x1",
                "y1",
                "x2",
                "y2",
                "name",
                "instance",
                "sem_id",
                "prob",
            ]
        )
        for frame_id, frame in enumerate(pipeline_result.frames):
            sensor = frame.metadata.get("source_name") or "unknown"
            device = frame.metadata.get("device_name") or "unknown"
            img_width = frame.metadata.get("image_width") or 0
            img_height = frame.metadata.get("image_height") or 0
            for detection in frame.detections_2d:
                writer.writerow(
                    [
                        frame.timestamp_ns,
                        frame_id,
                        sensor,
                        device,
                        img_width,
                        img_height,
                        f"{float(detection.xyxy[0]):.2f}",
                        f"{float(detection.xyxy[1]):.2f}",
                        f"{float(detection.xyxy[2]):.2f}",
                        f"{float(detection.xyxy[3]):.2f}",
                        detection.label,
                        detection.instance_id if detection.instance_id is not None else -1,
                        detection.sem_id if detection.sem_id is not None else -1,
                        f"{float(detection.score):.6f}",
                    ]
                )
