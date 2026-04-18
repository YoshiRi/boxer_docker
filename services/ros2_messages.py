from __future__ import annotations

from typing import Any


def frame_result_to_detection2d_array(
    frame_result,
    *,
    frame_id: str,
) -> dict[str, Any]:
    metadata = frame_result.metadata
    return {
        "header": {
            "stamp_ns": frame_result.timestamp_ns,
            "frame_id": frame_id,
        },
        "detections": [
            {
                "label": detection.label,
                "sem_id": detection.sem_id if detection.sem_id is not None else -1,
                "instance_id": detection.instance_id
                if detection.instance_id is not None
                else -1,
                "score": float(detection.score),
                "x1": float(detection.xyxy[0]),
                "y1": float(detection.xyxy[1]),
                "x2": float(detection.xyxy[2]),
                "y2": float(detection.xyxy[3]),
                "image_width": int(metadata.get("image_width") or 0),
                "image_height": int(metadata.get("image_height") or 0),
                "sensor": metadata.get("source_name") or "unknown",
                "device": metadata.get("device_name") or "unknown",
            }
            for detection in frame_result.detections_2d
        ],
    }


def frame_result_to_detection3d_array(
    frame_result,
    *,
    frame_id: str,
) -> dict[str, Any]:
    metadata = frame_result.metadata
    return {
        "header": {
            "stamp_ns": frame_result.timestamp_ns,
            "frame_id": frame_id,
        },
        "detections": [
            {
                "label": detection.label,
                "sem_id": detection.sem_id if detection.sem_id is not None else -1,
                "instance_id": detection.instance_id
                if detection.instance_id is not None
                else -1,
                "score": float(detection.score),
                "center_xyz": [float(value) for value in detection.center_xyz],
                "quaternion_wxyz": [
                    float(value) for value in detection.quaternion_wxyz
                ],
                "size_xyz": [float(value) for value in detection.size_xyz],
                "source_sensor": metadata.get("source_name") or "unknown",
                "source_device": metadata.get("device_name") or "unknown",
            }
            for detection in frame_result.detections_3d
        ],
    }


def frame_result_to_track3d_array(
    frame_result,
    *,
    frame_id: str,
) -> dict[str, Any]:
    return {
        "header": {
            "stamp_ns": frame_result.timestamp_ns,
            "frame_id": frame_id,
        },
        "tracks": [
            {
                "track_id": track.track_id,
                "support_count": track.support_count if track.support_count is not None else 0,
                "missed_count": track.missed_count if track.missed_count is not None else 0,
                "accumulated_weight": float(track.accumulated_weight)
                if track.accumulated_weight is not None
                else 0.0,
                "detection": {
                    "label": track.detection.label,
                    "sem_id": track.detection.sem_id
                    if track.detection.sem_id is not None
                    else -1,
                    "instance_id": track.detection.instance_id
                    if track.detection.instance_id is not None
                    else -1,
                    "score": float(track.detection.score),
                    "center_xyz": [float(value) for value in track.detection.center_xyz],
                    "quaternion_wxyz": [
                        float(value) for value in track.detection.quaternion_wxyz
                    ],
                    "size_xyz": [float(value) for value in track.detection.size_xyz],
                },
            }
            for track in frame_result.tracks_3d
        ],
    }


def pipeline_result_to_job_summary(
    pipeline_result,
    *,
    job_id: str,
    input_mode: str,
    input_path: str,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    artifacts = pipeline_result.artifacts or []
    return {
        "header": {
            "stamp_ns": pipeline_result.frames[-1].timestamp_ns
            if pipeline_result.frames
            else 0,
            "frame_id": pipeline_result.sequence_name,
        },
        "job_id": job_id,
        "sequence_name": pipeline_result.sequence_name,
        "input_mode": input_mode,
        "input_path": input_path,
        "output_root": pipeline_result.output_root or "",
        "frames_processed": len(pipeline_result.frames),
        "duration_sec": float(duration_sec) if duration_sec is not None else 0.0,
        "artifact_names": [str(item.get("name", "")) for item in artifacts],
        "artifact_paths": [str(item.get("path", "")) for item in artifacts],
        "artifact_exists": [bool(item.get("exists", False)) for item in artifacts],
    }
