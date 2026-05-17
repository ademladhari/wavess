#!/usr/bin/env bash
set -euo pipefail

# Google Colab — clone wavess and run wmbench.
#
# In a Colab cell:
#   !bash /content/run_colab_wmbench.sh
#
# Or download and run:
#   !wget -q https://raw.githubusercontent.com/ademladhari/wavess/main/run_colab_wmbench.sh -O /content/run_colab_wmbench.sh
#   !bash /content/run_colab_wmbench.sh

REPO_URL="https://github.com/ademladhari/wavess.git"
REPO_DIR="${WAVESS_REPO_DIR:-/content/wavess}"

# If this script lives inside an already-cloned repo, use that tree instead of re-cloning.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/wmbench/run_benchmark.py" ]]; then
  REPO_DIR="$SCRIPT_DIR"
fi

if [[ ! -f "$REPO_DIR/wmbench/run_benchmark.py" ]]; then
  echo "[0/4] Cloning $REPO_URL -> $REPO_DIR"
  rm -rf "$REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
echo "Using repo: $REPO_DIR"

echo "[1/4] Installing combined requirements (dct + wmbench)..."
python -m pip install --upgrade pip -q
# compressai needs numpy<2; pin before other packages.
python -m pip install "numpy>=1.26.0,<2.0" -q
python -m pip install -r "wmbench/requirements_colab_combined.txt" -q

echo "[2/4] Paths (Google Drive MyDrive — mount Drive in Colab first)"
# Mount Drive in a cell before running this script:
#   from google.colab import drive
#   drive.mount("/content/drive")
DRIVE_ROOT="${WAVESS_DRIVE_ROOT:-/content/drive/MyDrive}"
IMAGES_DIR="${WAVESS_IMAGES:-$DRIVE_ROOT/wmbench/images}"
NEGATIVES_DIR="${WAVESS_NEGATIVES:-$DRIVE_ROOT/negative_500}"
OUTPUT_DIR="${WAVESS_OUTPUT:-$DRIVE_ROOT/wmbench_results}"

if [[ ! -d "/content/drive/MyDrive" ]] && [[ "$DRIVE_ROOT" == /content/drive/MyDrive* ]]; then
  echo "ERROR: Google Drive not mounted at /content/drive/MyDrive"
  echo "Run in Colab first:"
  echo "  from google.colab import drive"
  echo "  drive.mount('/content/drive')"
  exit 1
fi
if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "ERROR: images folder not found: $IMAGES_DIR"
  echo "Put originals under MyDrive/wmbench/images or set WAVESS_IMAGES=/path/to/images"
  exit 1
fi
if [[ ! -d "$NEGATIVES_DIR" ]]; then
  echo "ERROR: negatives folder not found: $NEGATIVES_DIR"
  echo "Put negatives under MyDrive/negative_500 or set WAVESS_NEGATIVES=/path/to/negatives"
  exit 1
fi
echo "  images:    $IMAGES_DIR"
echo "  negatives: $NEGATIVES_DIR"
echo "  output:    $OUTPUT_DIR"

echo "[3/4] GPU check"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "[4/4] Running wmbench benchmark..."
python "wmbench/run_benchmark.py" \
  --methods dct \
  --images "$IMAGES_DIR" \
  --negatives "$NEGATIVES_DIR" \
  --output "$OUTPUT_DIR" \
  --device cuda \
  --diffusion-attack-batch-size "${WAVESS_DIFF_BATCH:-4}" \
  --lpips-batch-size "${WAVESS_LPIPS_BATCH:-16}" \
  --resume

echo "Done. Results in: $OUTPUT_DIR"
