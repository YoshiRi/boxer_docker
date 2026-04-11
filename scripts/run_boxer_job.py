#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_boxer import build_arg_parser, run_with_args


def _sequence_name(input_path: str) -> str:
    return os.path.basename(input_path.rstrip("/"))


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
    return [
        {"name": name, "path": str(path), "exists": path.exists()}
        for name, path in artifacts
    ]


def build_job_manifest(args) -> dict[str, Any]:
    seq_name = _sequence_name(args.input)
    output_root = Path(os.path.expanduser(args.output_dir)) / seq_name
    return {
        "input": {
            "input_path": args.input,
            "output_dir": str(Path(os.path.expanduser(args.output_dir))),
            "write_name": args.write_name,
            "max_n": args.max_n,
            "start_n": args.start_n,
            "skip_n": args.skip_n,
            "track": args.track,
            "fuse": args.fuse,
            "camera": args.camera,
            "labels": args.labels,
            "force_cpu": args.force_cpu,
            "skip_viz": args.skip_viz,
        },
        "sequence_name": seq_name,
        "output_root": str(output_root),
        "artifacts": _artifact_records(output_root, args.write_name, args.track),
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
    run_with_args(args)
    return build_job_manifest(args)


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
