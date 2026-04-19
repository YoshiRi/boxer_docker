# Boxer ROS2 Message Design

## Goal

Define a message-oriented ROS2 boundary for `boxer_api` without forcing the
core inference package to depend on ROS-specific message classes.

The design separates:

- inference result objects in `boxer_api`
- ROS-facing serialization and message mapping in `services/`
- future custom `.msg` package work in a dedicated ROS workspace

## Design Principles

- Keep `boxer_api` ROS-agnostic.
- Publish per-frame results separately from end-of-job summaries.
- Preserve timestamps and frame identity on every published result.
- Use structured message shapes rather than a single ad-hoc JSON blob.

## Recommended ROS Topics

- `/boxer/detections_2d`
- `/boxer/detections_3d`
- `/boxer/tracks_3d`
- `/boxer/job_summary`
- `/boxer/debug_image`

## Recommended Custom Message Types

### `Detection2D.msg`

```text
std_msgs/Header header
string label
int32 sem_id
int32 instance_id
float32 score
float32 x1
float32 y1
float32 x2
float32 y2
uint32 image_width
uint32 image_height
string sensor
string device
```

### `Detection2DArray.msg`

```text
std_msgs/Header header
Detection2D[] detections
```

### `Detection3D.msg`

```text
std_msgs/Header header
string label
int32 sem_id
int32 instance_id
float32 score
geometry_msgs/Point center
geometry_msgs/Quaternion orientation
geometry_msgs/Vector3 size
string source_sensor
string source_device
```

### `Detection3DArray.msg`

```text
std_msgs/Header header
Detection3D[] detections
```

### `Track3D.msg`

```text
std_msgs/Header header
int32 track_id
Detection3D detection
int32 support_count
int32 missed_count
float32 accumulated_weight
```

### `Track3DArray.msg`

```text
std_msgs/Header header
Track3D[] tracks
```

### `JobSummary.msg`

```text
std_msgs/Header header
string job_id
string sequence_name
string input_mode
string input_path
string output_root
uint32 frames_processed
float32 duration_sec
string[] artifact_names
string[] artifact_paths
bool[] artifact_exists
```

## Transitional JSON Contract

Until a dedicated ROS message package exists, the repository publishes a
`std_msgs/String` JSON payload from `services/ros2_node.py`.

That JSON is structured as if it had already been split into:

- `detection_2d_array`
- `detection_3d_array`
- `track_3d_array`

The conversion helpers for those payloads live in `services/ros2_messages.py`.

## Responsibilities

### `boxer_api`

- compute 2D detections
- compute 3D detections
- compute tracking / batch results

### `services/ros2_messages.py`

- convert `FrameResult` and `PipelineResult` to ROS-oriented dictionaries
- provide a stable bridge before real custom ROS messages are introduced

### future ROS package

- own `.msg` files
- convert the dictionary contract to actual ROS message classes

## Current Scope

Implemented now:

- structured JSON conversion helpers for frame-level 2D/3D/track arrays
- a minimal ROS2 node that publishes the structured JSON result

Still missing:

- actual `.msg` files and package metadata
- publish to `vision_msgs` or custom `boxer_msgs`
- camera-info ingestion
- tracked result publishing beyond empty placeholders
- batch summary publishing from a ROS node
