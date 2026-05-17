#!/usr/bin/env bash
set -euo pipefail

# Run this from the repository root in Google Colab.
# Example:
#   !bash /content/waves/run_colab_wmbench.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "[1/3] Installing combined requirements..."
python -m pip install --upgrade pip
python -m pip install -r "wmbench/requirements_colab_combined.txt"

echo "[2/3] Set your paths below before running..."
# Update these paths for your Colab session / Drive mount.
IMAGES_DIR="/content/images"
NEGATIVES_DIR="/content/negatives"
OUTPUT_DIR="/content/wmbench_results"

echo "[3/3] Running wmbench benchmark..."
python "wmbench/run_benchmark.py" \
  --methods dct \
  --images "$IMAGES_DIR" \
  --negatives "$NEGATIVES_DIR" \
  --output "$OUTPUT_DIR" \
  --device cuda \
  --resume

echo "Done. Results in: $OUTPUT_DIR"
