#!/usr/bin/env bash
# Download Amazon Berkeley Objects (ABO) via AWS CLI (no credentials required).
# License: CC BY-NC 4.0 — https://amazon-berkeley-objects.s3.amazonaws.com/index.html
#
# Usage:
#   bash data/download_abo.sh               # full download (images + renders + metadata)
#   bash data/download_abo.sh --images-only # catalog images + metadata only (no renders)
#   bash data/download_abo.sh --renders-only # renders + metadata only
#
# Requires: aws CLI  (pip install awscli  or  brew install awscli)
# No AWS credentials needed — all data is public.

set -euo pipefail

IMAGES_ONLY=false
RENDERS_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --images-only)  IMAGES_ONLY=true ;;
    --renders-only) RENDERS_ONLY=true ;;
  esac
done

DEST="data/raw/abo"
mkdir -p "$DEST"

# ---- Metadata (always downloaded — small, needed for preprocessing) ----
echo "[ABO] Downloading listing metadata (~500 MB gzipped JSON-lines)..."
aws s3 cp --no-sign-request \
  s3://amazon-berkeley-objects/listings/metadata/ \
  "$DEST/metadata/" \
  --recursive \
  --exclude "*" \
  --include "*.json.gz"

# ---- Catalog images (small JPEGs, 512px long-edge) ----
if [[ "$RENDERS_ONLY" == "false" ]]; then
  echo "[ABO] Downloading catalog images (small, ~40 GB)..."
  aws s3 cp --no-sign-request \
    s3://amazon-berkeley-objects/images/small/ \
    "$DEST/images/small/" \
    --recursive \
    --exclude "*" \
    --include "*.jpg"
fi

# ---- Rendered images (3-D turntable renders with segmentation masks) ----
# Each item has up to 72 spin frames; we download only frame 0 (front-facing)
# to keep storage manageable (~15 GB for spin-0 PNGs).
# Remove the --include filter to get all 72 frames per item.
if [[ "$IMAGES_ONLY" == "false" ]]; then
  echo "[ABO] Downloading rendered images — spin-0 only (~15 GB)..."
  aws s3 cp --no-sign-request \
    s3://amazon-berkeley-objects/renders/png/ \
    "$DEST/renders/png/" \
    --recursive \
    --exclude "*" \
    --include "*/0000.png"

  echo "[ABO] Downloading render segmentation masks — spin-0 only..."
  aws s3 cp --no-sign-request \
    s3://amazon-berkeley-objects/renders/segmentation_masks/ \
    "$DEST/renders/segmentation_masks/" \
    --recursive \
    --exclude "*" \
    --include "*/0000.png"
fi

echo ""
echo "[ABO] Download complete."
echo "  Metadata : $DEST/metadata/"
echo "  Images   : $DEST/images/small/"
echo "  Renders  : $DEST/renders/"
echo ""
echo "Next step: python data/preprocess_abo.py"
