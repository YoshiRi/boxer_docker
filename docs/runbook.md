# Boxer Headless Docker Runbook

This runbook reproduces README Demo #1 inside Docker:

```bash
python run_boxer.py --input nym10_gen1 --max_n=90 --track
```

It uses host-mounted directories for checkpoints, sample data, and outputs so model assets stay outside the image.

## 1. Build the container

CPU-only image:

```bash
docker compose build boxer
```

CUDA-capable image:

```bash
docker compose --profile gpu build boxer-gpu
```

Notes:

- The GPU path assumes Docker is configured with NVIDIA Container Toolkit.
- The GPU service switches the PyTorch wheel index to CUDA wheels but does not vendor CUDA inside this repo.

## 2. Download checkpoints and sample data

Minimal bootstrap for Demo #1:

```bash
./scripts/bootstrap_boxer.sh
```

That downloads:

- checkpoints into `./ckpts/`
- `nym10_gen1` sample data into `./sample_data/nym10_gen1/`

Download all bundled Aria samples instead:

```bash
./scripts/bootstrap_boxer.sh --all-aria
```

Download only checkpoints:

```bash
./scripts/bootstrap_boxer.sh --ckpts-only
```

Download only sample data:

```bash
./scripts/bootstrap_boxer.sh --data-only --aria-seq nym10_gen1
```

## 3. Sanity-check imports inside the container

CPU:

```bash
docker compose run --rm boxer python -c "import torch, cv2, dill, tqdm, projectaria_tools; print('imports ok')"
```

GPU:

```bash
docker compose --profile gpu run --rm boxer-gpu python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Run Demo #1 in Docker

CPU:

```bash
docker compose run --rm boxer \
  python run_boxer.py --input nym10_gen1 --max_n=90 --track
```

GPU:

```bash
docker compose --profile gpu run --rm boxer-gpu \
  python run_boxer.py --input nym10_gen1 --max_n=90 --track
```

## 5. Output locations on the host

The container mounts the repository root at `/workspace`, so outputs written by Boxer land directly on the host filesystem under:

- `./output/nym10_gen1/boxer_3dbbs.csv`
- `./output/nym10_gen1/owl_2dbbs.csv`
- `./output/nym10_gen1/boxer_3dbbs_tracked.csv`
- `./output/nym10_gen1/boxer_viz_current.jpg`
- `./output/nym10_gen1/boxer_viz_final.mp4`
- `./output/nym10_gen1/boxer_viz/`

## 6. Rerun from scratch

To rerun the demo while keeping downloaded assets:

```bash
rm -rf output/nym10_gen1
docker compose run --rm boxer \
  python run_boxer.py --input nym10_gen1 --max_n=90 --track
```

To force a fresh asset download:

```bash
rm -rf ckpts sample_data/nym10_gen1
./scripts/bootstrap_boxer.sh
```

## 7. Troubleshooting

If the run fails before inference:

- Check that `ckpts/` contains all three upstream files.
- Check that `sample_data/nym10_gen1/` contains the five Aria files expected by `AriaLoader`.
- Verify `ffmpeg` exists in the container if MP4 generation fails.
- Verify `projectaria-tools` imports successfully if VRS loading fails early.

If the run is too slow on CPU:

- That is expected for Boxer inference on CPU.
- Use the `boxer-gpu` profile on a host with NVIDIA runtime support.

If CUDA fails:

- Confirm `docker run --gpus all ...` works on the host first.
- Rebuild the GPU image after changing `TORCH_INDEX_URL_GPU` or `TORCH_VERSION`.
