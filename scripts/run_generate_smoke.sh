#!/usr/bin/env bash
# Minimal SD-19 smoke for end-to-end generation.
#
# Usage:
#   bash scripts/run_generate_smoke.sh path/to/product.jpg
#   bash scripts/run_generate_smoke.sh path/to/product.jpg shopify
#
# Notes:
# - Defaults to the Shopify IP-Adapter checkpoint.
# - Runs the stable "no ControlNet" path by default. Add --with-controlnet
#   after the positional args if you want to try the experimental branch.
# - First run may download SDXL / VAE / CLIP assets from Hugging Face.

set -euo pipefail

PRODUCT_PATH="${1:-}"
PLATFORM="${2:-shopify}"
CONTROLNET_FLAG="--disable-controlnet"

if [[ -z "$PRODUCT_PATH" ]]; then
  echo "Usage: bash scripts/run_generate_smoke.sh path/to/product.jpg [shopify|etsy|ebay] [--with-controlnet]"
  exit 1
fi

if [[ "${3:-}" == "--with-controlnet" ]]; then
  CONTROLNET_FLAG=""
fi

if [[ ! -f "$PRODUCT_PATH" ]]; then
  echo "Input image not found: $PRODUCT_PATH"
  exit 1
fi

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv Python at $PYTHON_BIN"
  exit 1
fi

ADAPTER_CKPT="checkpoints/ip_adapter/${PLATFORM}/final"
if [[ ! -d "$ADAPTER_CKPT" ]]; then
  echo "Adapter checkpoint not found: $ADAPTER_CKPT"
  exit 1
fi

SAM2_CKPT="checkpoints/sam2_hiera_large.pt"
if [[ ! -f "$SAM2_CKPT" ]]; then
  echo "SAM2 checkpoint not found: $SAM2_CKPT"
  exit 1
fi

OUT_DIR="outputs/generate_smoke/${PLATFORM}"
mkdir -p "$OUT_DIR"

echo "=== SD-19 smoke ==="
echo "product:   $PRODUCT_PATH"
echo "platform:  $PLATFORM"
echo "adapter:   $ADAPTER_CKPT"
echo "output:    $OUT_DIR"

"$PYTHON_BIN" inference/generate.py \
  --product "$PRODUCT_PATH" \
  --platform "$PLATFORM" \
  --adapter ip_adapter \
  --adapter_ckpt "$ADAPTER_CKPT" \
  ${CONTROLNET_FLAG} \
  --device mps \
  --segmentation-device cpu \
  --dtype fp16 \
  --steps 8 \
  --height 512 \
  --width 512 \
  --scale 0.75 \
  --mask-output "$OUT_DIR/mask.png" \
  --control-output "$OUT_DIR/control.png" \
  --output "$OUT_DIR/generated.png"

echo "=== Done ==="
