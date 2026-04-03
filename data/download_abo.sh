#!/usr/bin/env bash
# Download Amazon Berkeley Objects (ABO) via AWS CLI (no credentials required).
# License: CC BY-NC 4.0 — https://amazon-berkeley-objects.s3.amazonaws.com/index.html
#
# Usage: bash data/download_abo.sh [--images-only]

set -euo pipefail

DEST="data/raw/abo"
mkdir -p "$DEST"

echo "[ABO] Downloading metadata..."
aws s3 cp --no-sign-request \
  s3://amazon-berkeley-objects/listings/metadata/ \
  "$DEST/metadata/" --recursive

echo "[ABO] Downloading images (spins-0 only for speed; remove --exclude to get all)..."
aws s3 cp --no-sign-request \
  s3://amazon-berkeley-objects/images/small/ \
  "$DEST/images/small/" --recursive

echo "[ABO] Done. Images saved to $DEST/images/"
