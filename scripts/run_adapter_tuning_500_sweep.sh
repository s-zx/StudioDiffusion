#!/usr/bin/env bash
set -euo pipefail

PROGRESS_LOG="results/adapter_tuning_500_live_progress.log"
LOG_DIR="logs/adapter_tuning_500"
FRESH=false

if [[ "${1:-}" == "--fresh" ]]; then
  FRESH=true
fi

mkdir -p "$LOG_DIR"

if [[ "$FRESH" == true ]]; then
  : > "$PROGRESS_LOG"
fi

CONFIGS=(
  configs/tuning/ip_adapter/ebay_lr2e-4_s500.yaml
  configs/tuning/ip_adapter/ebay_lr5e-5_s500.yaml
  configs/tuning/ip_adapter/ebay_tokens32_s500.yaml
  configs/tuning/ip_adapter/ebay_tokens8_s500.yaml
  configs/tuning/ip_adapter/ebay_proj512_s500.yaml
  configs/tuning/ip_adapter/ebay_proj2048_s500.yaml
  configs/tuning/ip_adapter/etsy_lr2e-4_s500.yaml
  configs/tuning/ip_adapter/etsy_lr5e-5_s500.yaml
  configs/tuning/ip_adapter/etsy_tokens32_s500.yaml
  configs/tuning/ip_adapter/etsy_tokens8_s500.yaml
  configs/tuning/ip_adapter/etsy_proj512_s500.yaml
  configs/tuning/ip_adapter/etsy_proj2048_s500.yaml
  configs/tuning/ip_adapter/shopify_lr2e-4_s500.yaml
  configs/tuning/ip_adapter/shopify_lr5e-5_s500.yaml
  configs/tuning/ip_adapter/shopify_tokens32_s500.yaml
  configs/tuning/ip_adapter/shopify_tokens8_s500.yaml
  configs/tuning/ip_adapter/shopify_proj512_s500.yaml
  configs/tuning/ip_adapter/shopify_proj2048_s500.yaml
  configs/tuning/lora/ebay_lr2e-4_s500.yaml
  configs/tuning/lora/ebay_lr5e-5_s500.yaml
  configs/tuning/lora/ebay_rank32_alpha32_s500.yaml
  configs/tuning/lora/ebay_rank8_alpha8_s500.yaml
  configs/tuning/lora/etsy_lr2e-4_s500.yaml
  configs/tuning/lora/etsy_lr5e-5_s500.yaml
  configs/tuning/lora/etsy_rank32_alpha32_s500.yaml
  configs/tuning/lora/etsy_rank8_alpha8_s500.yaml
  configs/tuning/lora/shopify_lr2e-4_s500.yaml
  configs/tuning/lora/shopify_lr5e-5_s500.yaml
  configs/tuning/lora/shopify_rank32_alpha32_s500.yaml
  configs/tuning/lora/shopify_rank8_alpha8_s500.yaml
)

{
  printf '\n============================================================\n'
  printf 'ADAPTER TUNING 500-STEP SWEEP START date=%s\n' "$(date -Is)"
  printf 'runs=%s\n' "${#CONFIGS[@]}"
  printf 'steps_per_run=500\n'
  printf 'progress_log=%s\n' "$PROGRESS_LOG"
  printf 'skip_existing_final=true\n'
  printf '============================================================\n'
  printf 'RUN_ORDER\n'
  printf '%s\n' "${CONFIGS[@]}"
  printf '============================================================\n'
} >> "$PROGRESS_LOG"

for config in "${CONFIGS[@]}"; do
  adapter=$(basename "$(dirname "$config")")
  run_name=$(basename "$config" .yaml)
  checkpoint_dir="checkpoints/${adapter}/${run_name}"
  console_log="${LOG_DIR}/${adapter}_${run_name}.console.log"

  if [[ -d "${checkpoint_dir}/final" ]]; then
    {
      printf 'SKIP adapter=%s run=%s date=%s reason=final_exists\n' "$adapter" "$run_name" "$(date -Is)"
      printf 'checkpoint_dir=%s\n' "$checkpoint_dir"
    } >> "$PROGRESS_LOG"
    continue
  fi

  {
    printf '\n============================================================\n'
    printf 'START adapter=%s run=%s date=%s\n' "$adapter" "$run_name" "$(date -Is)"
    printf 'config=%s\n' "$config"
    printf 'checkpoint_dir=%s\n' "$checkpoint_dir"
    printf 'train_log=%s/train.log\n' "$checkpoint_dir"
    printf 'console_log=%s\n' "$console_log"
    printf '============================================================\n'
  } >> "$PROGRESS_LOG"

  if [[ "$adapter" == "ip_adapter" ]]; then
    launcher="scripts/train_ip_adapter.sh"
  else
    launcher="scripts/train_lora.sh"
  fi

  if { time bash "$launcher" "$config"; } > "$console_log" 2>&1; then
    {
      printf 'DONE adapter=%s run=%s date=%s\n' "$adapter" "$run_name" "$(date -Is)"
      if [[ -f "${checkpoint_dir}/train.log" ]]; then
        printf 'SUMMARY adapter=%s run=%s ' "$adapter" "$run_name"
        awk '/val_loss=/{line=$0} END{if(line) print line; else print "val_loss=NA"}' "${checkpoint_dir}/train.log"
      fi
    } >> "$PROGRESS_LOG"
  else
    status=$?
    {
      printf 'FAILED adapter=%s run=%s date=%s exit_code=%s\n' "$adapter" "$run_name" "$(date -Is)" "$status"
      printf 'console_log=%s\n' "$console_log"
    } >> "$PROGRESS_LOG"
    exit "$status"
  fi
done

{
  printf '\n============================================================\n'
  printf 'ADAPTER TUNING 500-STEP SWEEP COMPLETE date=%s\n' "$(date -Is)"
  printf '============================================================\n'
} >> "$PROGRESS_LOG"