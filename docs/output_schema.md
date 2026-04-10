# Boxer Output Schema

This document records the actual output artifacts and CSV schemas emitted by the current codebase. It is intended to make later ROS integration straightforward.

## Output root

Default output root:

- `output/`

Per-sequence output directory:

- `output/<sequence_name>/`

For Demo #1:

- `output/nym10_gen1/`

## Artifact inventory

Produced by `run_boxer.py` in headless mode:

- `boxer_3dbbs.csv`
  Per-frame 3D detections written during inference.
- `owl_2dbbs.csv`
  Per-frame 2D detections written during inference.
- `boxer_3dbbs_tracked.csv`
  Final active tracked 3D boxes, written only when `--track` is enabled and active tracks exist at shutdown.
- `boxer_viz_current.jpg`
  Last rendered visualization frame.
- `boxer_viz_final.mp4`
  Video assembled from intermediate JPG frames with `ffmpeg`.
- `boxer_viz/boxer_viz_*.jpg`
  Intermediate rendered frames used to create the MP4.

## `boxer_3dbbs.csv`

Writer implementation:

- `utils/file_io.py`, `ObbCsvWriter2`

Header:

```text
time_ns,tx_world_object,ty_world_object,tz_world_object,qw_world_object,qx_world_object,qy_world_object,qz_world_object,scale_x,scale_y,scale_z,name,instance,sem_id,prob
```

Column semantics:

- `time_ns`
  Frame timestamp in nanoseconds from the input loader.
- `tx_world_object`, `ty_world_object`, `tz_world_object`
  Object translation in world coordinates, meters.
- `qw_world_object`, `qx_world_object`, `qy_world_object`, `qz_world_object`
  Object orientation quaternion in world coordinates.
- `scale_x`, `scale_y`, `scale_z`
  3D box dimensions along object-local axes, meters.
- `name`
  Semantic label string carried from the text prompt / detector mapping.
- `instance`
  Instance ID. For normal inference rows this is usually not a stable cross-frame ID.
- `sem_id`
  Integer semantic ID assigned from the text-label mapping.
- `prob`
  Confidence. In `run_boxer.py` this is the mean of the retained 2D score and the 3D score.

Notes for ROS integration:

- This file is the best base for emitting 3D detection messages per frame.
- It is world-frame data already, not camera-frame box parameters.

## `boxer_3dbbs_tracked.csv`

Writer implementation:

- `run_boxer.py` final tracker flush using `ObbCsvWriter2`

Schema:

- Same header and columns as `boxer_3dbbs.csv`

Behavior details:

- Written once, after inference completes.
- Contains only active tracks that remain at the end of the run.
- `instance` is overwritten with `track_id`.
- `prob` is rounded to two decimals before writing.
- `time_ns` is currently written as `0` for all rows in this file.

ROS implication:

- This file is useful as a compact summary of the final tracker state.
- It is not a full per-frame tracked history stream.

## `owl_2dbbs.csv`

Writer implementation:

- `utils/file_io.py`, `save_bb2d_csv`

Header:

```text
time_ns,frame_id,sensor,device,img_width,img_height,x1,y1,x2,y2,name,instance,sem_id,prob
```

Column semantics:

- `time_ns`
  Frame timestamp in nanoseconds.
- `frame_id`
  Zero-based iteration index inside `run_boxer.py`.
- `sensor`
  Loader camera name such as `rgb`, `slaml`, or `slamr`.
- `device`
  Loader device name such as `Aria Gen 1` or `Aria Gen 2`.
- `img_width`, `img_height`
  Image size for interpreting the pixel coordinates.
- `x1`, `y1`, `x2`, `y2`
  Pixel coordinates in standard XYXY image convention.
- `name`
  Label string.
- `instance`
  Always `-1` in the current implementation.
- `sem_id`
  Integer semantic ID matching the in-memory label map.
- `prob`
  2D detector confidence score.

Notes for ROS integration:

- This file provides image-space detections aligned with the rendered RGB view.
- It can be paired with `time_ns` for synchronization with future ROS topics.

## Visualization outputs

Current implementation details:

- Per-frame images are JPGs, not PNGs.
- MP4 creation depends on `ffmpeg`.
- `boxer_viz_current.jpg` is overwritten every frame.

ROS implication:

- These artifacts are useful for offline debugging, not as canonical machine-readable outputs.

## Known inconsistencies

- README text mentions `outputs/...` in one place, but code writes to `output/...`.
- README examples mention `boxer_viz_current.png`, but code writes `boxer_viz_current.jpg`.
