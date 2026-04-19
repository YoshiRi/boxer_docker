# boxer_api Design

## Goal

This repository already contains the core Boxer model, the OWLv2 detector wrapper, dataset and stream input adapters, and several application entrypoints such as `run_boxer.py` and `webui/server.py`.

What it does not have yet is a stable, implementation-oriented API boundary for real integration work.

The goal of `boxer_api/` is to introduce that boundary without forking the codebase into a separate repository and without duplicating model or loader logic.

The intended result is:

- a reusable Python API for Boxer inference
- a stable set of typed input/output objects
- separation between inference core and CLI / file-writing orchestration
- a migration path where existing tools (`run_boxer.py`, batch jobs, web UI) can gradually depend on the API layer

## Why Keep This Inside The Current Repository

The current repository is still the source of truth for:

- model loading
- checkpoint paths
- input source resolution
- tracking and fusion behavior
- output schema

Creating a separate repository now would introduce parallel implementations and make every upstream change harder to propagate.

For that reason, the API layer should be added as a new internal package:

- `boxer_api/`

This keeps the API close to the existing implementation while making the public integration surface explicit.

## Non-Goals

The first phase should **not** attempt to:

- redesign BoxerNet internals
- replace all existing loaders
- remove `run_boxer.py`
- turn the API into a network service
- expose every internal debug artifact through the public surface

The first phase is about establishing a clean Python integration boundary.

## Core Design Principle

Separate the system into three layers:

1. `InputAdapter`
   Converts files, datasets, streams, or ROS inputs into a common frame representation.

2. `InferenceEngine`
   Runs 2D detection and/or 3D lifting on a single frame.

3. `Pipeline`
   Coordinates multi-frame execution, optional tracking, fusion, artifact writing, and manifest generation.

The existing repository already contains pieces of each layer, but they are currently coupled together most heavily in `run_boxer.py`.

## Proposed Package Layout

```text
boxer_api/
├── __init__.py
├── types.py
├── config.py
├── engine.py
├── pipeline.py
├── adapters.py
├── detectors.py
└── artifacts.py
```

### `types.py`

Defines stable data structures used by external callers.

Planned types:

- `FrameInput`
- `Detection2D`
- `Detection3D`
- `Track3D`
- `FrameResult`
- `PipelineResult`

These should be narrow, explicit, and decoupled from CLI argument parsing.

### `config.py`

Defines API-facing configuration objects.

Planned config groups:

- `DetectorConfig`
- `BoxerConfig`
- `TrackingConfig`
- `PipelineConfig`

This avoids exposing a large CLI-shaped parameter surface directly to API callers.

### `engine.py`

Contains the reusable single-frame inference core.

Target responsibilities:

- load BoxerNet
- optionally load OWLv2
- accept a normalized `FrameInput`
- accept external 2D boxes or produce them internally
- return typed 2D/3D detections

This is the most important new boundary.

### `pipeline.py`

Contains multi-frame orchestration.

Target responsibilities:

- iterate over frame inputs
- call `InferenceEngine`
- optionally update tracker state
- optionally run fusion
- return typed results plus references to any artifacts

This should become the internal implementation target for:

- `run_boxer.py`
- `scripts/run_boxer_job.py`
- parts of `webui/server.py`

### `adapters.py`

Bridges existing `input_sources/` implementations into the API layer.

This should avoid reimplementing loaders. Instead, it should translate the existing loader/source outputs into `FrameInput`.

### `detectors.py`

Provides detector-facing adapters and factories so the engine does not talk to `owl.owl_wrapper.OwlWrapper` directly everywhere.

### `artifacts.py`

Bridges typed inference results to the repository's current artifact outputs:

- CSV writers
- visualization frames
- manifest records

This keeps file output behavior available without forcing all callers to think in terms of files.

## Public API Shape

Two public entry styles are desirable.

### 1. Low-Level Frame API

For callers that already have ROIs / detections:

```python
result = engine.infer_frame(
    frame=frame_input,
    detections_2d=detections_2d,
)
```

This is the right interface for production systems that already own 2D detection or ROI generation.

### 2. Full Frame API

For callers that want Boxer to perform 2D detection internally:

```python
result = engine.infer_frame(
    frame=frame_input,
    labels=["chair", "table", "lamp"],
)
```

Internally this will:

- run OWLv2
- map labels
- run BoxerNet

### 3. Sequence / Batch API

For callers that want orchestration:

```python
result = pipeline.run_sequence(
    frames=frame_iterable,
    config=pipeline_config,
)
```

This is the right interface for:

- CLI
- batch jobs
- web UI
- offline integration tests

## Input Model

The API must not assume that `image + ROI` is sufficient for all uses.

The current model path uses more than that:

- image
- camera intrinsics
- world/rig pose
- sparse or semi-dense 3D points
- optional external 2D boxes
- timestamp

The stable API input should therefore be centered on a `FrameInput` object with fields along these lines:

- `image_bgr`
- `camera`
- `pose_world_rig`
- `sparse_points_world`
- `timestamp_ns`
- `rotated`
- `source_name`
- `device_name`

If a simplified single-image path is needed later, it should be provided as a convenience wrapper on top of the full frame input, not as the primary internal abstraction.

## Output Model

The public result should not expose raw repository-internal tensor wrappers by default.

Instead it should provide typed, structured results such as:

- list of `Detection2D`
- list of `Detection3D`
- optional list of `Track3D`
- optional timing/debug information

Internally, the API may still use repository-specific tensor wrappers, but the stable outer interface should not depend on them.

## Migration Strategy

This work should be done incrementally.

### Step 1

Create `boxer_api/` with:

- package init
- typed data structures
- configuration objects

No behavior change yet.

### Step 2

Extract a reusable single-frame inference boundary from `run_boxer.py`.

This means isolating:

- 2D detection input/output handling
- BoxerNet forward pass
- confidence filtering
- semantic label mapping

The result should be callable without invoking the full CLI flow.

### Step 3

Add a sequence/pipeline layer that can process an iterable of `FrameInput`.

Initially this can be minimal and omit visualization or file outputs.

### Step 4

Bridge existing artifact outputs:

- CSV
- MP4 / JPG generation
- manifest generation

This is the point where `scripts/run_boxer_job.py` can start depending on `boxer_api.pipeline`.

### Step 5

Refactor `run_boxer.py` to become a CLI front-end over `boxer_api`.

The CLI should remain compatible while delegating more logic to the new package.

## Expected Benefits

- cleaner integration path for downstream applications
- easier testing of inference behavior independent of CLI
- clearer separation between stable API and research/demo scripts
- reduced risk of duplicating logic across CLI, jobs, and web UI

## Immediate Next Action

Start with Step 1:

- add `boxer_api/`
- define public types and configs
- keep the initial scope narrow and stable
