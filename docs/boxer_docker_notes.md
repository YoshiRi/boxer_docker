# Boxer Docker Notes

This note records the Docker-specific findings from inspecting the codebase, not just the README.

## Runtime dependencies confirmed from code

For the headless `run_boxer.py` path used by Demo #1, the required Python packages are:

- `torch`
- `numpy`
- `opencv-python` or `opencv-python-headless`
- `tqdm`
- `projectaria-tools`
- `Pillow`
- `dill`

Why these matter in code:

- `run_boxer.py` imports `cv2`, `numpy`, `torch`, `tqdm`, `BoxerNet`, `AriaLoader`, and video helpers.
- `loaders/aria_loader.py` imports `projectaria_tools.core.data_provider`.
- `loaders/omni_loader.py` and `loaders/ca_loader.py` import `PIL.Image`.
- `utils/video.py` shells out to `ffmpeg` to create the final MP4.
- `dill` is listed in the upstream install instructions and is part of the acceptance import set, even though the current headless path does not import it directly.

Packages that are only needed for the interactive viewers are intentionally left out of the Docker runtime:

- `moderngl`
- `moderngl-window`
- `imgui-bundle`

## System packages confirmed from code

Headless Demo #1 needs these non-Python components:

- `ffmpeg`
  `utils/video.py` calls it to build `boxer_viz_final.mp4`.
- `bash`, `curl` or `wget`, `ca-certificates`
  Needed for bootstrap and download flows.
- `libgomp1`
  Useful for PyTorch/OpenMP runtime compatibility on slim Debian images.

GUI/OpenGL packages are out of scope for this first iteration because `run_boxer.py` does not import the viewer stack.

## Output-path findings from code

There is a README/code inconsistency worth keeping explicit:

- The README references `outputs/nym10_gen1/` in one place.
- The code uses `utils/demo_utils.py`:
  `EVAL_PATH = os.path.join(_REPO_ROOT, "output")`

So the actual default output root is:

- `output/<sequence_name>/`

For Demo #1 with `--input nym10_gen1 --track`, the actual output directory is:

- `output/nym10_gen1/`

Artifacts written by code:

- `output/nym10_gen1/boxer_3dbbs.csv`
- `output/nym10_gen1/owl_2dbbs.csv`
- `output/nym10_gen1/boxer_3dbbs_tracked.csv`
- `output/nym10_gen1/boxer_viz_current.jpg`
- `output/nym10_gen1/boxer_viz_final.mp4`
- `output/nym10_gen1/boxer_viz/*.jpg`

Notable details:

- The current-frame image is `.jpg`, not `.png`.
- Intermediate visualization frames are `.jpg`.
- The tracked CSV is only written at the end of the run if active tracks remain.

## Aria sample-data layout confirmed from code

For bare sequence names like `nym10_gen1`, `run_boxer.py` resolves the input under:

- `sample_data/<sequence>/`

`loaders/aria_loader.py` then expects at least:

- `main.vrs`
- `closed_loop_trajectory.csv`
- `online_calibration.jsonl`
- `semidense_points.csv.gz`
- `semidense_observations.csv.gz`

## CPU and CUDA approach

The container is kept simple:

- CPU build uses PyTorch CPU wheels by default.
- GPU build swaps the PyTorch wheel index to CUDA wheels and relies on host NVIDIA Container Toolkit support.

This keeps the Dockerfile maintainable while still allowing a CUDA-capable path when the host is prepared for it.

## Known caveats

- `projectaria-tools` wheel availability can vary by Python/platform. The Dockerfile uses Python 3.12 because the repo targets 3.12 in `pyproject.toml`, but wheel availability should still be validated in your environment.
- CUDA execution depends on host driver compatibility with the chosen PyTorch CUDA wheel.
- No ROS packaging or ROS message adapters are included yet; the documentation below keeps output schemas explicit so that can be added cleanly later.
