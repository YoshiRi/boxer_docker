# Boxer ROS Bridge Plan

This document defines the batch I/O boundary for later ROS integration without turning Boxer itself into a ROS package yet.

## Wrapper entrypoint

Use `scripts/run_boxer_job.py` as the stable Python entrypoint.

It supports the same core inference flags as `run_boxer.py`, but adds one ROS-friendly behavior:

- it can be imported and called as a Python function
- it emits a JSON manifest alongside the normal CSV and visualization artifacts
- the manifest includes run timing plus per-artifact size and CSV row counts

Example:

```bash
python scripts/run_boxer_job.py \
  --input nym10_gen1 \
  --max_n 90 \
  --track \
  --manifest output/nym10_gen1/job_manifest.json
```

Python usage:

```python
from scripts.run_boxer_job import run_boxer_job

result = run_boxer_job(
    input_path="nym10_gen1",
    max_n=90,
    track=True,
)
```

## Input contract

Minimum required input:

- `input_path`
  Boxer sequence identifier or path, same semantics as `run_boxer.py --input`

Optional control fields:

- `output_dir`
- `max_n`
- `start_n`
- `skip_n`
- `track`
- `fuse`
- `labels`
- `camera`
- `force_cpu`
- `skip_viz`

This is intentionally a batch job contract, not a frame-by-frame streaming API.

## Output contract

Primary machine-readable outputs:

- `output/<sequence>/boxer_3dbbs.csv`
- `output/<sequence>/owl_2dbbs.csv`
- `output/<sequence>/boxer_3dbbs_tracked.csv` when `track=True`
- `output/<sequence>/job_manifest.json`

The manifest JSON records:

- job status and timing metadata
- requested input parameters
- resolved per-sequence output directory
- expected artifact paths
- existence flags for each artifact
- artifact sizes in bytes
- CSV row counts where applicable

This gives a single file a ROS-side launcher can inspect before publishing results downstream.

Example manifest shape:

```json
{
  "job": {
    "status": "completed",
    "run_started_at_utc": "2026-04-11T01:23:45+00:00",
    "manifest_created_at_utc": "2026-04-11T01:24:03+00:00",
    "duration_sec": 18.217
  },
  "input": {
    "input_path": "nym10_gen1",
    "track": true
  },
  "sequence_name": "nym10_gen1",
  "output_root": "output/nym10_gen1",
  "artifacts": [
    {
      "name": "boxer_3dbbs_csv",
      "path": "output/nym10_gen1/boxer_3dbbs.csv",
      "exists": true,
      "size_bytes": 463669,
      "row_count": 2201
    }
  ]
}
```

## Suggested ROS mapping

Recommended boundary for a future ROS 2 package:

- service or action request
  Input: sequence path, labels, thresholds, runtime flags
- job worker
  Calls `run_boxer_job(...)` in-process
- published outputs
  `owl_2dbbs.csv` -> `vision_msgs/Detection2DArray`
  `boxer_3dbbs.csv` -> custom or derived 3D detection message
  `boxer_3dbbs_tracked.csv` -> tracked object topic
  `boxer_viz_current.jpg` -> debug image topic

## Non-goals for this stage

- real-time camera subscription
- direct ROS message emission inside Boxer core
- ROS package layout, launch files, or colcon integration
