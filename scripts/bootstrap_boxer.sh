#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${BOXER_CKPT_DIR:-${REPO_ROOT}/ckpts}"
DATA_DIR="${BOXER_SAMPLE_DATA_DIR:-${REPO_ROOT}/sample_data}"
HF_MODEL_BASE="${BOXER_MODEL_BASE_URL:-https://huggingface.co/facebook/boxer/resolve/main}"
HF_DATA_BASE="${BOXER_DATA_BASE_URL:-https://huggingface.co/datasets/facebook/boxer/resolve/main}"

CKPT_FILES=(
  "boxernet_hw960in4x6d768-wssxpf9p.ckpt"
  "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
  "owlv2-base-patch16-ensemble.pt"
)

ARIA_FILES=(
  "main.vrs"
  "closed_loop_trajectory.csv"
  "online_calibration.jsonl"
  "semidense_observations.csv.gz"
  "semidense_points.csv.gz"
)

ALL_ARIA_SEQS=("hohen_gen1" "nym10_gen1" "cook0_gen2")
ARIA_SEQS=("nym10_gen1")
DOWNLOAD_CKPTS=1
DOWNLOAD_ARIA=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --ckpts-only           Download checkpoints only.
  --data-only            Download Aria sample data only.
  --aria-seq NAME        Download a specific Aria sample sequence. Repeatable.
  --all-aria             Download all bundled Aria sample sequences.
  --ckpt-dir PATH        Override checkpoint directory.
  --data-dir PATH        Override sample-data directory.
  -h, --help             Show this help.

Defaults:
  checkpoints: enabled
  Aria sample data: nym10_gen1 only

Environment overrides:
  BOXER_CKPT_DIR, BOXER_SAMPLE_DATA_DIR,
  BOXER_MODEL_BASE_URL, BOXER_DATA_BASE_URL
EOF
}

download_file() {
  local url="$1"
  local dest="$2"

  if [[ -f "$dest" ]]; then
    echo "Already exists: $dest"
    return
  fi

  mkdir -p "$(dirname "$dest")"
  echo "Downloading: $url"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
  else
    echo "Missing downloader: install wget or curl." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpts-only)
      DOWNLOAD_CKPTS=1
      DOWNLOAD_ARIA=0
      shift
      ;;
    --data-only)
      DOWNLOAD_CKPTS=0
      DOWNLOAD_ARIA=1
      shift
      ;;
    --aria-seq)
      ARIA_SEQS+=("$2")
      shift 2
      ;;
    --all-aria)
      ARIA_SEQS=("${ALL_ARIA_SEQS[@]}")
      shift
      ;;
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if (( DOWNLOAD_CKPTS )); then
  echo "==> Downloading checkpoints into $CKPT_DIR"
  mkdir -p "$CKPT_DIR"
  for file in "${CKPT_FILES[@]}"; do
    download_file "${HF_MODEL_BASE}/${file}" "${CKPT_DIR}/${file}"
  done
fi

if (( DOWNLOAD_ARIA )); then
  echo "==> Downloading Aria sample data into $DATA_DIR"
  mkdir -p "$DATA_DIR"
  declare -A seen=()
  for seq in "${ARIA_SEQS[@]}"; do
    [[ -n "$seq" ]] || continue
    if [[ -n "${seen[$seq]:-}" ]]; then
      continue
    fi
    seen[$seq]=1
    for file in "${ARIA_FILES[@]}"; do
      download_file "${HF_DATA_BASE}/${seq}/${file}" "${DATA_DIR}/${seq}/${file}"
    done
  done
fi

echo "Bootstrap complete."
echo "  checkpoints: $CKPT_DIR"
echo "  sample_data: $DATA_DIR"
