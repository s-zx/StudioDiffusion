#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

CONFIGS=(
  configs/tuning/ip_adapter/*_s250.yaml
  configs/tuning/lora/*_s250.yaml
)

bash scripts/run_adapter_tuning_stage.sh --stage adapter_quality_screen_250 "$@" "${CONFIGS[@]}"