#!/usr/bin/env bash
# Train LoRA adapter for a given platform.
#
# Usage:
#   bash scripts/train_lora.sh shopify
#   bash scripts/train_lora.sh etsy
#   bash scripts/train_lora.sh ebay

set -euo pipefail

TARGET="${1:-shopify}"

if [[ "$TARGET" == *.yaml ]]; then
  CONFIG="$TARGET"
  PLATFORM="$(basename "$TARGET" .yaml)"
else
  PLATFORM="$TARGET"
  CONFIG="configs/lora/${PLATFORM}.yaml"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "=== Training LoRA for platform: $PLATFORM ==="

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Run as module so `from adapters.lora.model import ...` resolves from repo root.
PYTHONPATH="${PWD}:${PYTHONPATH:-}" accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  -m adapters.lora.train \
  --config "$CONFIG"

echo "=== Done. Checkpoint saved to checkpoints/lora/$PLATFORM/final ==="
