#!/usr/bin/env python3

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from input_sources import infer_sequence_name
from run_boxer import build_arg_parser, run_with_args


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        # Skip header row.
        next(handle, None)
        return sum(1 for _ in handle)


def _artifact_records(output_root: Path, write_name: str, track: bool) -> list[dict[str, Any]]:
    artifacts = [
        ("boxer_3dbbs_csv", output_root / f"{write_name}_3dbbs.csv"),
        ("owl_2dbbs_csv", output_root / "owl_2dbbs.csv"),
        ("boxer_viz_current_jpg", output_root / f"{write_name}_viz_current.jpg"),
        ("boxer_viz_final_mp4", output_root / f"{write_name}_viz_final.mp4"),
    ]
    if track:
        artifacts.append(
            ("boxer_3dbbs_tracked_csv", output_root / f"{write_name}_3dbbs_tracked.csv")
        )

    records = []
    for name, path in artifacts:
        exists = path.exists()
        record = {
            "name": name,
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
        }
        row_count = _count_csv_rows(path)
        if row_count is not None:
            record["row_count"] = row_count
        records.append(record)
    return records


def build_job_manifest(args, *, run_started_at: str | None = None, duration_sec: float | None = None) -> dict[str, Any]:
    seq_name = infer_sequence_name(args)
    output_root = Path(os.path.expanduser(args.output_dir)) / seq_name
    artifacts = _artifact_records(output_root, args.write_name, args.track)
    return {
        "job": {
            "status": "completed",
            "run_started_at_utc": run_started_at,
            "manifest_created_at_utc": _utc_now_iso(),
            "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
        },
        "input": {
            "input_path": args.input,
            "output_dir": str(Path(os.path.expanduser(args.output_dir))),
            "write_name": args.write_name,
            "input_mode": args.input_mode,
            "input_glob": args.input_glob,
            "input_metadata": args.input_metadata,
            "max_n": args.max_n,
            "start_n": args.start_n,
            "skip_n": args.skip_n,
            "track": args.track,
            "fuse": args.fuse,
            "camera": args.camera,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "camera_fx": args.camera_fx,
            "camera_fy": args.camera_fy,
            "camera_cx": args.camera_cx,
            "camera_cy": args.camera_cy,
            "frame_period_ns": args.frame_period_ns,
            "start_time_ns": args.start_time_ns,
            "stream_name": args.stream_name,
            "labels": args.labels,
            "force_cpu": args.force_cpu,
            "skip_viz": args.skip_viz,
        },
        "sequence_name": seq_name,
        "output_root": str(output_root),
        "artifacts": artifacts,
    }


def run_boxer_job(**kwargs) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args([])
    if "input_path" in kwargs:
        if "input" in kwargs:
            raise TypeError("Use either 'input' or 'input_path', not both")
        kwargs["input"] = kwargs.pop("input_path")
    for key, value in kwargs.items():
        if not hasattr(args, key):
            raise TypeError(f"Unknown Boxer job argument: {key}")
        setattr(args, key, value)
    run_started_at = _utc_now_iso()
    start_time = time.perf_counter()
    run_with_args(args)
    duration_sec = time.perf_counter() - start_time
    return build_job_manifest(
        args,
        run_started_at=run_started_at,
        duration_sec=duration_sec,
    )


def main(argv=None):
    parser = build_arg_parser()
    parser.description = "Batch wrapper around run_boxer.py for downstream integration."
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional JSON manifest path. Defaults to <output>/<sequence>/job_manifest.json.",
    )
    args = parser.parse_args(argv)
    boxer_kwargs = vars(args).copy()
    manifest_path = boxer_kwargs.pop("manifest")
    manifest = run_boxer_job(**boxer_kwargs)

    if manifest_path is None:
        manifest_path = os.path.join(manifest["output_root"], "job_manifest.json")

    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"==> Wrote job manifest to {manifest_file}")


if __name__ == "__main__":
    main()
