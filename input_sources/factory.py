from __future__ import annotations

import os
import re
from dataclasses import dataclass

from loaders.ca_loader import CALoader
from loaders.omni_loader import OMNI3D_DATASETS, OmniLoader
from loaders.scannet_loader import ScanNetLoader
from utils.demo_utils import SAMPLE_DATA_PATH

from .file_sequence_source import ImageSequenceFrameSource
from .frame_source import sanitize_sequence_name
from .loader_source import LoaderFrameSource
from .opencv_source import OpenCvCaptureFrameSource
from .ros2_source import Ros2ImageFrameSource

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(slots=True)
class ResolvedFrameSource:
    source: object
    kind: str
    sequence_name: str


def _directory_has_image_files(input_path: str, image_glob: str) -> bool:
    patterns = [pattern.strip() for pattern in image_glob.split(",") if pattern.strip()]
    allowed_exts = {
        os.path.splitext(pattern.lstrip("*"))[1].lower()
        for pattern in patterns
        if pattern.startswith("*.")
    }
    if not allowed_exts:
        allowed_exts = IMAGE_EXTENSIONS
    with os.scandir(input_path) as entries:
        for entry in entries:
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in allowed_exts:
                return True
    return False


def add_frame_source_args(parser) -> None:
    parser.add_argument(
        "--input_mode",
        type=str,
        default="auto",
        choices=["auto", "aria", "ca1m", "omni3d", "scannet", "file", "cv2", "ros2"],
        help="Input source type. Defaults to auto-detect for existing dataset paths.",
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        default="*.jpg,*.jpeg,*.png",
        help="Comma-separated image globs for --input_mode=file.",
    )
    parser.add_argument(
        "--input_metadata",
        type=str,
        default=None,
        help="Optional JSON metadata path for file/cv2/ros2 sources.",
    )
    parser.add_argument("--camera_width", type=int, default=None, help="Static pinhole width override for file/cv2/ros2 sources.")
    parser.add_argument("--camera_height", type=int, default=None, help="Static pinhole height override for file/cv2/ros2 sources.")
    parser.add_argument("--camera_fx", type=float, default=None, help="Static pinhole fx for file/cv2/ros2 sources.")
    parser.add_argument("--camera_fy", type=float, default=None, help="Static pinhole fy for file/cv2/ros2 sources.")
    parser.add_argument("--camera_cx", type=float, default=None, help="Static pinhole cx for file/cv2/ros2 sources.")
    parser.add_argument("--camera_cy", type=float, default=None, help="Static pinhole cy for file/cv2/ros2 sources.")
    parser.add_argument("--frame_period_ns", type=int, default=100_000_000, help="Synthetic timestamp spacing for file/cv2/ros2 sources.")
    parser.add_argument("--start_time_ns", type=int, default=0, help="Synthetic starting timestamp for file/cv2/ros2 sources.")
    parser.add_argument("--stream_name", type=str, default=None, help="Optional logical sequence name override for file/cv2/ros2 sources.")
    parser.add_argument("--ros_compressed", action="store_true", help="Use sensor_msgs/CompressedImage when --input_mode=ros2.")
    parser.add_argument("--ros_node_name", type=str, default="boxer_input_source", help="ROS2 node name for --input_mode=ros2.")
    parser.add_argument("--ros_queue_size", type=int, default=8, help="ROS2 subscription queue size for --input_mode=ros2.")


def infer_input_mode(args) -> str:
    if args.input_mode != "auto":
        return args.input_mode
    input_path = os.path.expanduser(args.input)
    if os.path.isfile(input_path):
        suffix = os.path.splitext(input_path)[1].lower()
        if suffix in IMAGE_EXTENSIONS:
            return "file"
        if suffix in VIDEO_EXTENSIONS:
            return "cv2"
    if os.path.isdir(input_path):
        if os.path.exists(os.path.join(input_path, "main.vrs")):
            return "aria"
        if _directory_has_image_files(input_path, args.input_glob):
            return "file"
    if bool(re.search(r"scene\d{4}_\d{2}", args.input)) or "/scannet/" in args.input:
        return "scannet"
    if args.input in OMNI3D_DATASETS:
        return "omni3d"
    if args.input.startswith("ca1m"):
        return "ca1m"
    return "aria"


def infer_sequence_name(args, input_mode: str | None = None) -> str:
    input_mode = input_mode or infer_input_mode(args)
    if args.stream_name:
        return sanitize_sequence_name(args.stream_name)
    if input_mode in {"scannet", "aria"}:
        return os.path.basename(os.path.expanduser(args.input).rstrip("/"))
    if input_mode == "file":
        input_path = os.path.expanduser(args.input).rstrip("/")
        base_name = os.path.basename(input_path) or os.path.basename(os.path.dirname(input_path))
        return sanitize_sequence_name(base_name)
    if input_mode in {"omni3d", "ca1m"}:
        return args.input
    if input_mode == "cv2":
        expanded = os.path.expanduser(args.input)
        if os.path.exists(expanded):
            return sanitize_sequence_name(os.path.basename(expanded), default="cv2_input")
        return sanitize_sequence_name(f"cv2_{args.input}", default="cv2_input")
    if input_mode == "ros2":
        return sanitize_sequence_name(args.input.strip("/"), default="ros2_stream")
    return os.path.basename(args.input.rstrip("/"))


def _resolve_aria_root(input_path: str) -> str:
    remote_root = input_path
    if not os.path.isabs(remote_root) and not os.path.exists(remote_root):
        sample = os.path.join(SAMPLE_DATA_PATH, remote_root)
        legacy = os.path.expanduser(os.path.join("~/boxy_data", remote_root))
        if os.path.exists(sample):
            remote_root = sample
        elif os.path.exists(legacy):
            remote_root = legacy
    return remote_root


def resolve_input_source(args) -> ResolvedFrameSource:
    input_mode = infer_input_mode(args)
    seq_name = infer_sequence_name(args, input_mode=input_mode)

    if input_mode == "scannet":
        loader = ScanNetLoader(
            scene_dir=args.input,
            annotation_path=os.path.join(SAMPLE_DATA_PATH, "scannet", "full_annotations.json"),
            skip_frames=args.skip_n,
            max_frames=args.max_n,
            start_frame=args.start_n,
        )
        seq_name = loader.scene_id
        return ResolvedFrameSource(
            source=LoaderFrameSource(loader, kind="scannet", sequence_name=seq_name),
            kind="scannet",
            sequence_name=seq_name,
        )

    if input_mode == "omni3d":
        loader = OmniLoader(
            dataset_name=args.input,
            split="val",
            max_images=args.max_n,
            skip_images=args.skip_n,
        )
        return ResolvedFrameSource(
            source=LoaderFrameSource(loader, kind="omni3d", sequence_name=seq_name),
            kind="omni3d",
            sequence_name=seq_name,
        )

    if input_mode == "ca1m":
        loader = CALoader(
            seq_name,
            start_frame=args.start_n,
            skip_frames=args.skip_n,
            max_frames=args.max_n,
            resize=(args.detector_hw, args.detector_hw),
        )
        return ResolvedFrameSource(
            source=LoaderFrameSource(loader, kind="ca1m", sequence_name=seq_name),
            kind="ca1m",
            sequence_name=seq_name,
        )

    if input_mode == "aria":
        from loaders.aria_loader import AriaLoader

        remote_root = _resolve_aria_root(args.input)
        loader = AriaLoader(
            remote_root,
            camera=args.camera,
            with_traj=True,
            with_sdp=True,
            with_obb=args.gt2d,
            pinhole=args.pinhole,
            resize=None,
            unrotate=False,
            skip_n=args.skip_n,
            max_n=args.max_n,
            start_n=args.start_n,
        )
        return ResolvedFrameSource(
            source=LoaderFrameSource(loader, kind="aria", sequence_name=seq_name),
            kind="aria",
            sequence_name=seq_name,
        )

    if input_mode == "file":
        source = ImageSequenceFrameSource(
            args.input,
            image_glob=args.input_glob,
            metadata_path=args.input_metadata,
            start_frame=max(0, args.start_n - 1),
            skip_frames=args.skip_n,
            max_frames=args.max_n,
            camera_name=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fx=args.camera_fx,
            fy=args.camera_fy,
            cx=args.camera_cx,
            cy=args.camera_cy,
            frame_period_ns=args.frame_period_ns,
            start_time_ns=args.start_time_ns,
            sequence_name=seq_name,
        )
        return ResolvedFrameSource(source=source, kind="file", sequence_name=seq_name)

    if input_mode == "cv2":
        source = OpenCvCaptureFrameSource(
            args.input,
            start_frame=max(0, args.start_n - 1),
            skip_frames=args.skip_n,
            max_frames=args.max_n,
            camera_name=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fx=args.camera_fx,
            fy=args.camera_fy,
            cx=args.camera_cx,
            cy=args.camera_cy,
            frame_period_ns=args.frame_period_ns,
            start_time_ns=args.start_time_ns,
            sequence_name=seq_name,
        )
        return ResolvedFrameSource(source=source, kind="cv2", sequence_name=seq_name)

    if input_mode == "ros2":
        source = Ros2ImageFrameSource(
            args.input,
            compressed=args.ros_compressed,
            node_name=args.ros_node_name,
            queue_size=args.ros_queue_size,
            max_frames=args.max_n,
            camera_name=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            fx=args.camera_fx,
            fy=args.camera_fy,
            cx=args.camera_cx,
            cy=args.camera_cy,
            frame_period_ns=args.frame_period_ns,
            start_time_ns=args.start_time_ns,
            sequence_name=seq_name,
        )
        return ResolvedFrameSource(source=source, kind="ros2", sequence_name=seq_name)

    raise ValueError(f"Unsupported input_mode: {input_mode}")
