# Boxer Input Interface

This document describes the flexible frame-source interface used by `run_boxer.py` and `scripts/run_boxer_job.py`.

## Goal

The original code path bound Boxer inference directly to dataset-specific loaders. That was sufficient for Aria and benchmark datasets, but it made it awkward to reuse the same inference loop for:

- a single image or image directory replay
- OpenCV-backed video or webcam input
- ROS 2 image topics

The new design isolates frame ingestion from inference so new sources can feed Boxer without duplicating the core runner.

## Core abstraction

The interface lives under `input_sources/`.

- `FrameSourceInfo`
  Declares the resolved source kind, sequence name, logical camera, and device identity.
- `BaseFrameSource`
  Pull-based iterator contract returning Boxer datum dictionaries.
- `PushFrameSource`
  Queue-backed variant for live producers such as ROS 2 callbacks.
- `LoaderFrameSource`
  Thin adapter that wraps the existing dataset loaders so old and new inputs share the same runner path.

Each yielded datum follows the same keys the model already expects:

- `img0`
- `cam0`
- `T_world_rig0`
- `sdp_w`
- `time_ns0`
- `rotated0`

## Implemented adapters

### Dataset-backed

- `aria`
- `ca1m`
- `omni3d`
- `scannet`

These preserve the existing behavior and are wrapped by `LoaderFrameSource`.

### `file`

Implemented by `ImageSequenceFrameSource`.

Supported inputs:

- a single image file
- a directory of images discovered via `--input_glob`
- optional per-frame metadata JSON via `--input_metadata`

Behavior:

- synthesizes timestamps from `--start_time_ns` and `--frame_period_ns` when metadata does not provide them
- uses an identity pose when external pose metadata is absent
- builds a static pinhole camera from CLI intrinsics or the first image size

### `cv2`

Implemented by `OpenCvCaptureFrameSource`.

Supported inputs:

- video file path
- webcam index such as `0`
- any source OpenCV `VideoCapture` can open

Behavior:

- uses identity pose
- builds a static pinhole camera from the first decoded frame or CLI intrinsics
- supports `start_n`, `skip_n`, and `max_n`

### `ros2`

Implemented by `Ros2ImageFrameSource`.

Supported messages:

- `sensor_msgs/Image` with `rgb8`, `bgr8`, or `mono8`
- `sensor_msgs/CompressedImage` when `--ros_compressed` is set

Behavior:

- subscribes on a background spin loop and pushes decoded frames into a queue
- uses header timestamps when present, otherwise synthesizes them from `--start_time_ns` and `--frame_period_ns`
- uses identity pose and static pinhole intrinsics from CLI values or first-frame dimensions

Current limitation:

- runtime validation for ROS 2 has not been completed in this repo snapshot
- no camera-info topic ingestion yet
- no ROS-side output publisher yet

## Runner integration

`run_boxer.py` now does source resolution through `input_sources.factory`:

1. resolve `--input_mode`
2. derive a stable sequence name
3. build the matching frame source
4. feed the existing Boxer inference loop

This keeps:

- output layout
- CSV writing
- tracking
- headless visualization

unchanged across input modes.

## CLI additions

New shared flags:

- `--input_mode`
- `--input_glob`
- `--input_metadata`
- `--stream_name`
- `--frame_period_ns`
- `--start_time_ns`
- `--camera_width`
- `--camera_height`
- `--camera_fx`
- `--camera_fy`
- `--camera_cx`
- `--camera_cy`
- `--ros_compressed`
- `--ros_node_name`
- `--ros_queue_size`

## Validation status

Validated in Docker GPU:

- `file` mode using `cook0_gpu_viz_current.jpg`
- `cv2` mode using `cook0_gpu_viz_final.mp4`

Validated by Python import/compile:

- `ros2` adapter module loads without importing ROS dependencies at import time
- runtime ROS 2 execution still depends on an environment with `rclpy` and `sensor_msgs`

## Next steps

If this should become a production live-stream interface, the next missing pieces are:

1. camera-info ingestion for ROS 2
2. explicit output publisher interfaces instead of file-only artifacts
3. stronger lifecycle cleanup for long-lived live sources
4. end-to-end ROS 2 runtime validation in a matching container or host environment
