#!/usr/bin/env bash
# Download LAION-Aesthetics v2 (score >= 6.0) using img2dataset.
#
# Uses the public HuggingFace dataset  laion/aesthetics_6plus  which is
# pre-filtered to CLIP aesthetic score >= 6.0 (~600K images, ~300 GB raw).
#
# To keep storage manageable we download a capped subset via --max_shard_count.
# Remove that flag to get the full dataset.
#
# Requires:
#   pip install img2dataset huggingface-hub
#
# Usage:
#   bash data/download_laion.sh               # default: 50K images
#   bash data/download_laion.sh --full        # full ~600K images (warning: large)
#   bash data/download_laion.sh --n_shards 10 # custom shard count

set -euo pipefail

DEST="data/raw/laion_aesthetics"
PARQUET_DIR="$DEST/parquet"
FULL=false
N_SHARDS=5          # each shard ≈ 1,000 images ≈ 10K images for 10 shards

for arg in "$@"; do
  case "$arg" in
    --full)            FULL=true ;;
    --n_shards)        shift; N_SHARDS="$1" ;;
    --n_shards=*)      N_SHARDS="${arg#*=}" ;;
  esac
done

mkdir -p "$PARQUET_DIR" "$DEST/images"

# ---- Step 1: Download parquet metadata from HuggingFace ----
echo "[LAION] Downloading parquet index from laion/aesthetics_6plus ..."
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="laion/aesthetics_6plus",
    repo_type="dataset",
    local_dir="$PARQUET_DIR",
    ignore_patterns=["*.arrow", "*.lock", "README*"],
)
print("[LAION] Parquet files downloaded.")
EOF

# ---- Step 2: Download images via img2dataset ----
if [[ "$FULL" == "true" ]]; then
  SHARD_FLAG=""
  echo "[LAION] Downloading FULL dataset (this may take many hours and ~300 GB)..."
else
  SHARD_FLAG="--max_shard_count $N_SHARDS"
  echo "[LAION] Downloading capped subset (${N_SHARDS} shards ≈ $((N_SHARDS * 1000)) images)..."
fi

img2dataset \
  --url_list          "$PARQUET_DIR" \
  --input_format      "parquet" \
  --url_col           "url" \
  --caption_col       "caption" \
  --output_folder     "$DEST/images" \
  --output_format     "webdataset" \
  --image_size        512 \
  --min_image_size    256 \
  --resize_mode       "keep_ratio" \
  --number_sample_per_shard 1000 \
  --processes_count   8 \
  --thread_count      64 \
  --save_additional_columns '["similarity","watermark_score","aesthetic_score"]' \
  --enable_wandb      false \
  ${SHARD_FLAG}

echo ""
echo "[LAION] Download complete."
echo "  Parquet metadata : $PARQUET_DIR"
echo "  Images (webdataset tars) : $DEST/images/"
echo ""
echo "Note: LAION images are supplementary. Platform curation (SD-16) primarily"
echo "      uses ABO + DeepFashion2. LAION can be added as an extra --sources arg."
