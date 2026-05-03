#!/usr/bin/env bash
# Train the educational IP-Adapter v2 path.
#
# Usage:
#   bash scripts/train_ip_adapter_v2.sh shopify
#   bash scripts/train_ip_adapter_v2.sh etsy
#   bash scripts/train_ip_adapter_v2.sh ebay
#   bash scripts/train_ip_adapter_v2.sh _smoke
#   bash scripts/train_ip_adapter_v2.sh configs/ip_adapter/shopify_v2.yaml

set -euo pipefail

TARGET="${1:-shopify}"

if [[ "$TARGET" == *.yaml ]]; then
  CONFIG="$TARGET"
elif [[ "$TARGET" == *_v2 ]]; then
  CONFIG="configs/ip_adapter/${TARGET}.yaml"
else
  CONFIG="configs/ip_adapter/${TARGET}_v2.yaml"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== Training IP-Adapter v2 with config: $CONFIG ==="

PYTHONPATH="${PWD}:${PYTHONPATH:-}" time accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  -m adapters.ip_adapter.train_v2 \
  --config "$CONFIG"

echo "=== Done. Checkpoints are under checkpoints/ip_adapter_v2/ ==="
