#!/usr/bin/env bash
# Train IP-Adapter for a given platform.
#
# Usage:
#   bash scripts/train_ip_adapter.sh shopify
#   bash scripts/train_ip_adapter.sh etsy
#   bash scripts/train_ip_adapter.sh ebay

set -euo pipefail

PLATFORM="${1:-shopify}"
CONFIG="configs/ip_adapter/${PLATFORM}.yaml"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "=== Training IP-Adapter for platform: $PLATFORM ==="

accelerate launch \
  --mixed_precision fp16 \
  --num_processes 1 \
  adapters/ip_adapter/train.py \
  --config "$CONFIG"

echo "=== Done. Checkpoint saved to checkpoints/ip_adapter/$PLATFORM/final ==="
