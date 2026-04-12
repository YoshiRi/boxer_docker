# Boxer Docker Hands-On

This document captures the practical Docker workflow validated in this repo, including the extra steps needed when host bind mounts end up owned by `nobody:nogroup`.

## Scope

This hands-on covers:

- building the CPU and GPU Docker images
- downloading Boxer checkpoints and sample data
- running a validated headless demo
- downloading and running an additional sample sequence locally
- fixing ownership of Docker-generated artifacts on the host
- understanding what "stream input" means in the current codebase

It does not cover ROS package wiring, launch files, or direct ROS topic subscription.

## 1. Prerequisites

Host assumptions:

- Docker is installed and usable
- for GPU runs, the host exposes an NVIDIA GPU to Docker
- this repo is checked out locally

Useful checks:

```bash
docker info
nvidia-smi
```

In the validated environment, `boxer-gpu` saw:

- `torch.cuda.is_available() == True`
- device name `NVIDIA GeForce RTX 3060 Laptop GPU`

## 2. Standard Docker Demo

CPU:

```bash
make build
make bootstrap
make demo1
```

GPU:

```bash
make build-gpu
make bootstrap
make demo1-gpu
```

Saved CPU log:

```bash
make demo1-log
```

Expected host-side outputs for the default demo:

- `output/nym10_gen1/boxer_3dbbs.csv`
- `output/nym10_gen1/owl_2dbbs.csv`
- `output/nym10_gen1/boxer_3dbbs_tracked.csv`
- `output/nym10_gen1/boxer_viz_current.jpg`
- `output/nym10_gen1/boxer_viz_final.mp4`

## 3. Additional Sample Demo: `cook0_gen2`

The repo default bootstrap path is designed around `sample_data/`. In this environment, that directory had previously been created by Docker as `nobody:nogroup`, so writing a new sample there failed.

The reliable workaround is:

1. download the extra sample into a separate writable host directory
2. mount that directory into the container explicitly
3. write outputs into the normal `output/` bind mount

### 3.1 Download the sample locally

```bash
mkdir -p downloaded_sample_data
./scripts/bootstrap_boxer.sh \
  --data-only \
  --data-dir "$PWD/downloaded_sample_data" \
  --aria-seq cook0_gen2
```

Expected downloaded files:

- `downloaded_sample_data/cook0_gen2/main.vrs`
- `downloaded_sample_data/cook0_gen2/closed_loop_trajectory.csv`
- `downloaded_sample_data/cook0_gen2/online_calibration.jsonl`
- `downloaded_sample_data/cook0_gen2/semidense_observations.csv.gz`
- `downloaded_sample_data/cook0_gen2/semidense_points.csv.gz`

### 3.2 Run inference in Docker

GPU replay run:

```bash
docker compose --profile gpu run --rm \
  -v "$PWD/downloaded_sample_data:/opt/boxer/downloaded_sample_data" \
  boxer-gpu \
  bash -lc "python run_boxer.py \
    --input /opt/boxer/downloaded_sample_data/cook0_gen2 \
    --max_n=90 \
    --track \
    --output_dir output/downloaded_samples \
    --write_name cook0_gpu \
    | tee /opt/boxer/logs/cook0_gen2_gpu.log"
```

Observed outputs from the validated `cook0_gen2` run:

- `output/downloaded_samples/cook0_gen2/cook0_gpu_3dbbs.csv`
- `output/downloaded_samples/cook0_gen2/owl_2dbbs.csv`
- `output/downloaded_samples/cook0_gen2/cook0_gpu_3dbbs_tracked.csv`
- `output/downloaded_samples/cook0_gen2/cook0_gpu_viz_current.jpg`
- `output/downloaded_samples/cook0_gen2/cook0_gpu_viz_final.mp4`
- `logs/cook0_gen2_gpu.log`

Observed artifact counts:

- `cook0_gpu_3dbbs.csv`: 1891 rows
- `owl_2dbbs.csv`: 2067 rows
- `cook0_gpu_3dbbs_tracked.csv`: 51 rows
- `cook0_gpu_viz/`: 81 JPG frames

Why 81 JPGs instead of 90:

- the first few frames were skipped with `time misalignment`
- this is consistent with the Aria trajectory/image timestamp alignment behavior at the beginning of the selected range

### 3.3 Open the final visualization

Representative files:

- `output/downloaded_samples/cook0_gen2/cook0_gpu_viz_current.jpg`
- `output/downloaded_samples/cook0_gen2/cook0_gpu_viz_final.mp4`

## 4. Fixing Host Ownership

If Docker-generated files show up as `nobody:nogroup`, fix them from inside the container while the bind mounts are attached.

Validated command:

```bash
docker compose --profile gpu run --rm \
  -v "$PWD/downloaded_sample_data:/opt/boxer/downloaded_sample_data" \
  boxer-gpu \
  bash -lc "chown -R 1000:1000 /opt/boxer/output /opt/boxer/logs /opt/boxer/downloaded_sample_data"
```

After that, the following should belong to the host user again:

- `output/`
- `logs/`
- `downloaded_sample_data/`

If this happens repeatedly, the structural fix is to run the Docker service as the host UID/GID instead of root.

## 5. What Counts as "Stream Input" Here

There are two different meanings:

### 5.1 Recorded stream replay

This is supported now.

`run_boxer.py` opens a recorded sequence and consumes frames in order. For Project Aria inputs, that means it is already replaying the camera stream stored in `main.vrs`.

In other words, these commands are already "stream replay" demos:

```bash
make demo1
make demo1-gpu
```

and:

```bash
docker compose --profile gpu run --rm \
  -v "$PWD/downloaded_sample_data:/opt/boxer/downloaded_sample_data" \
  boxer-gpu \
  python run_boxer.py \
    --input /opt/boxer/downloaded_sample_data/cook0_gen2 \
    --max_n=90 \
    --track
```

The current Aria loader reads a recorded VRS stream sequentially and feeds Boxer frame by frame.

### 5.2 Live stream subscription

This is not supported yet.

Current constraints:

- `run_boxer.py` accepts a dataset name or local sequence path, not a live socket/topic/camera handle
- `scripts/run_boxer_job.py` is explicitly batch-oriented
- `docs/ros_bridge_plan.md` lists real-time subscription as a non-goal for the current stage

## 6. Minimal Path to a Non-ROS Stream Demo

If the goal is "show Boxer consuming a stream-like source" without implementing ROS yet, the shortest path is:

1. Treat a recorded Aria VRS as the stream source.
2. Run a long-lived Boxer process against that VRS.
3. Continuously read `boxer_viz_current.jpg` from the output directory as the latest frame-level visualization.
4. Use the generated CSVs or manifest as the machine-readable result stream.

That gives you a stream replay demo without changing Boxer core.

Concrete command:

```bash
docker compose --profile gpu run --rm \
  -v "$PWD/downloaded_sample_data:/opt/boxer/downloaded_sample_data" \
  boxer-gpu \
  bash -lc "python scripts/run_boxer_job.py \
    --input /opt/boxer/downloaded_sample_data/cook0_gen2 \
    --max_n 90 \
    --track \
    --write_name cook0_stream \
    --output_dir output/stream_demo \
    --manifest output/stream_demo/cook0_gen2/job_manifest.json"
```

Resulting files to watch:

- `output/stream_demo/cook0_gen2/cook0_stream_viz_current.jpg`
- `output/stream_demo/cook0_gen2/cook0_stream_3dbbs.csv`
- `output/stream_demo/cook0_gen2/owl_2dbbs.csv`
- `output/stream_demo/cook0_gen2/cook0_stream_3dbbs_tracked.csv`
- `output/stream_demo/cook0_gen2/job_manifest.json`

This is still batch over a finite replay window, but it is the closest supported demo to a stream-processing workflow.

## 7. What Must Be Added for True Live Streaming

For an actual live stream demo, the missing piece is an adapter layer between the live source and Boxer.

Minimum work items:

1. a frame ingestion loop that receives image frames plus pose/calibration metadata
2. a live loader interface matching what `run_boxer.py` currently expects from `AriaLoader`
3. a long-lived inference loop that preserves tracker state across frames
4. an output publisher for the latest image, 2D detections, 3D boxes, and tracks

The cleanest architecture is:

- keep Boxer core inference logic reusable
- add a new live-input runner alongside `run_boxer.py`
- let ROS or any non-ROS source feed that live runner later

Until that exists, the recommended demo is recorded stream replay from VRS.
