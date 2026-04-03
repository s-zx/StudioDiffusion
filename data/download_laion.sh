#!/usr/bin/env bash
# Download a subset of LAION-Aesthetics (score >= 6.0) using img2dataset.
#
# Requires: pip install img2dataset
# Usage: bash data/download_laion.sh

set -euo pipefail

DEST="data/raw/laion_aesthetics"
PARQUET_DIR="$DEST/parquet"
mkdir -p "$PARQUET_DIR"

echo "[LAION] Downloading parquet metadata files..."
# LAION-Aesthetics v2 parquet index (public, HuggingFace)
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="laion/laion2B-en-aesthetic",
    repo_type="dataset",
    local_dir="data/raw/laion_aesthetics/parquet",
    ignore_patterns=["*.arrow"],
)
EOF

echo "[LAION] Downloading images via img2dataset (aesthetic_score >= 6.0)..."
img2dataset \
  --url_list "$PARQUET_DIR" \
  --input_format "parquet" \
  --url_col "URL" \
  --caption_col "TEXT" \
  --output_folder "$DEST/images" \
  --output_format "webdataset" \
  --image_size 512 \
  --min_image_size 256 \
  --number_sample_per_shard 1000 \
  --processes_count 8 \
  --thread_count 64 \
  --save_additional_columns '["aesthetic_score"]' \
  --enable_wandb false

echo "[LAION] Done. Images saved to $DEST/images/"
