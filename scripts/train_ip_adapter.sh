#!/usr/bin/env bash
# Train IP-Adapter for a given platform.
#
# Usage:
#   bash scripts/train_ip_adapter.sh shopify
#   bash scripts/train_ip_adapter.sh etsy
#   bash scripts/train_ip_adapter.sh ebay

set -euo pipefail

TARGET="${1:-shopify}"

if [[ "$TARGET" == *.yaml ]]; then
  CONFIG="$TARGET"
  PLATFORM="$(basename "$TARGET" .yaml)"
else
  PLATFORM="$TARGET"
  CONFIG="configs/ip_adapter/${PLATFORM}.yaml"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "=== Training IP-Adapter for platform: $PLATFORM ==="

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# PYTHONPATH puts the repo root on sys.path so `from adapters.ip_adapter.model import ...`
# in train.py resolves. (pip install -e . is broken by the pyproject.toml build-backend
# typo, and accelerate launch passes the script as a path, which only adds the script's
# own directory to sys.path — not the repo root.)
# MPS fallback handles a small number of ops not yet implemented on Apple Silicon.
PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  adapters/ip_adapter/train.py \
  --config "$CONFIG"

echo "=== Done. Checkpoint saved to checkpoints/ip_adapter/$PLATFORM/final ==="
