#!/usr/bin/env bash
# Train LoRA adapter for a given platform.
#
# Usage:
#   bash scripts/train_lora.sh shopify
#   bash scripts/train_lora.sh etsy
#   bash scripts/train_lora.sh ebay

set -euo pipefail

PLATFORM="${1:-shopify}"
CONFIG="configs/lora/${PLATFORM}.yaml"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "=== Training LoRA for platform: $PLATFORM ==="

accelerate launch \
  --mixed_precision fp16 \
  --num_processes 1 \
  adapters/lora/train.py \
  --config "$CONFIG"

echo "=== Done. Checkpoint saved to checkpoints/lora/$PLATFORM/final ==="
