#!/usr/bin/env bash
set -euo pipefail

# Google Colab — clone wavess, run on fast local disk, sync results to Drive.
#
# Before running, mount Drive in Colab:
#   from google.colab import drive
#   drive.mount("/content/drive")
#
# Then:
#   !bash /content/wavess/run_colab_wmbench.sh
#
# Optional env:
#   WAVESS_METHODS=dwt|dct
#   WAVESS_DRIVE_ROOT=/content/drive/MyDrive
#   WAVESS_DIFF_BATCH=8
#   WAVESS_LPIPS_BATCH=32
#   WAVESS_SKIP_RINSE4X=1

REPO_URL="https://github.com/ademladhari/wavess.git"
REPO_DIR="${WAVESS_REPO_DIR:-/content/wavess}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/wmbench/run_benchmark.py" ]]; then
  REPO_DIR="$SCRIPT_DIR"
fi

if [[ ! -f "$REPO_DIR/wmbench/run_benchmark.py" ]]; then
  echo "[0/6] Cloning $REPO_URL -> $REPO_DIR"
  rm -rf "$REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
echo "Using repo: $REPO_DIR"

echo "[1/6] Installing dependencies..."
python -m pip install --upgrade pip -q
python -m pip install "numpy>=1.26.0,<2.0" -q
python -m pip install -r "wmbench/requirements_colab_combined.txt" -q
python -m pip install PyWavelets -q

DRIVE_ROOT="${WAVESS_DRIVE_ROOT:-/content/drive/MyDrive}"
DRIVE_IMAGES="${WAVESS_DRIVE_IMAGES:-$DRIVE_ROOT/wmbench/images}"
DRIVE_NEGATIVES="${WAVESS_DRIVE_NEGATIVES:-$DRIVE_ROOT/negative_500}"
DRIVE_OUTPUT="${WAVESS_DRIVE_OUTPUT:-$DRIVE_ROOT/wmbench_results}"

LOCAL_ROOT="${WAVESS_LOCAL_ROOT:-/content/wmbench_data}"
LOCAL_IMAGES="$LOCAL_ROOT/images"
LOCAL_NEGATIVES="$LOCAL_ROOT/negatives"
LOCAL_OUTPUT="${WAVESS_LOCAL_OUTPUT:-/content/wmbench_output}"

METHODS="${WAVESS_METHODS:-dwt}"
DIFF_BATCH="${WAVESS_DIFF_BATCH:-8}"
LPIPS_BATCH="${WAVESS_LPIPS_BATCH:-32}"
EXTRA_ARGS=()
if [[ "${WAVESS_SKIP_RINSE4X:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-rinse4xdiff)
fi

if [[ ! -d "/content/drive/MyDrive" ]] && [[ "$DRIVE_ROOT" == /content/drive/MyDrive* ]]; then
  echo "ERROR: Google Drive not mounted. Run: drive.mount('/content/drive')"
  exit 1
fi
if [[ ! -d "$DRIVE_IMAGES" ]]; then
  echo "ERROR: Drive images not found: $DRIVE_IMAGES"
  exit 1
fi
if [[ ! -d "$DRIVE_NEGATIVES" ]]; then
  echo "ERROR: Drive negatives not found: $DRIVE_NEGATIVES"
  exit 1
fi

echo "[2/6] Copy Drive -> local disk (skip if already present)..."
mkdir -p "$LOCAL_IMAGES" "$LOCAL_NEGATIVES" "$LOCAL_OUTPUT"
if [[ ! "$(ls -A "$LOCAL_IMAGES" 2>/dev/null)" ]]; then
  echo "  copying images -> $LOCAL_IMAGES"
  cp -a "$DRIVE_IMAGES/." "$LOCAL_IMAGES/"
else
  echo "  images already on local disk: $LOCAL_IMAGES"
fi
if [[ ! "$(ls -A "$LOCAL_NEGATIVES" 2>/dev/null)" ]]; then
  echo "  copying negatives -> $LOCAL_NEGATIVES"
  cp -a "$DRIVE_NEGATIVES/." "$LOCAL_NEGATIVES/"
else
  echo "  negatives already on local disk: $LOCAL_NEGATIVES"
fi

echo "  local images:    $LOCAL_IMAGES"
echo "  local negatives: $LOCAL_NEGATIVES"
echo "  local output:    $LOCAL_OUTPUT"
echo "  drive output:    $DRIVE_OUTPUT"

echo "[3/6] GPU check"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "[4/6] Running benchmark on local disk (fast I/O)..."
python "wmbench/run_benchmark.py" \
  --methods $METHODS \
  --images "$LOCAL_IMAGES" \
  --negatives "$LOCAL_NEGATIVES" \
  --output "$LOCAL_OUTPUT" \
  --device cuda \
  --diffusion-attack-batch-size "$DIFF_BATCH" \
  --lpips-batch-size "$LPIPS_BATCH" \
  --resume \
  "${EXTRA_ARGS[@]}"

echo "[5/6] Syncing results local -> Google Drive..."
mkdir -p "$DRIVE_OUTPUT"
cp -a "$LOCAL_OUTPUT/." "$DRIVE_OUTPUT/"

echo "[6/6] Done."
echo "  Local work:  $LOCAL_OUTPUT"
echo "  Drive copy:  $DRIVE_OUTPUT"
