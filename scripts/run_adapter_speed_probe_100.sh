#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/tuning/ip_adapter/ebay_baseline_s100.yaml
  configs/tuning/ip_adapter/ebay_batch4_ga2_s100.yaml
  configs/tuning/ip_adapter/ebay_no_ckpt_s100.yaml
  configs/tuning/ip_adapter/ebay_workers4_s100.yaml
  configs/tuning/ip_adapter/ebay_image768_s100.yaml
  configs/tuning/lora/ebay_baseline_s100.yaml
  configs/tuning/lora/ebay_batch4_ga2_s100.yaml
  configs/tuning/lora/ebay_no_ckpt_s100.yaml
  configs/tuning/lora/ebay_workers4_s100.yaml
  configs/tuning/lora/ebay_image768_s100.yaml
)

bash scripts/run_adapter_tuning_stage.sh --stage adapter_speed_probe_100 "$@" "${CONFIGS[@]}"