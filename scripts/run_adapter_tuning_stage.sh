#!/usr/bin/env bash
set -euo pipefail

FRESH=false
CONTINUE_ON_ERROR=false
STAGE="adapter_tuning_stage"
CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH=true
      shift
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=true
      shift
      ;;
    *)
      CONFIGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "usage: $0 [--fresh] --stage <name> <config.yaml> [config.yaml ...]" >&2
  exit 2
fi

PROGRESS_LOG="results/${STAGE}_live_progress.log"
LOG_DIR="logs/${STAGE}"
mkdir -p "$(dirname "$PROGRESS_LOG")" "$LOG_DIR"

if [[ "$FRESH" == true ]]; then
  : > "$PROGRESS_LOG"
fi

{
  printf '\n============================================================\n'
  printf 'ADAPTER TUNING STAGE START stage=%s date=%s\n' "$STAGE" "$(date -Is)"
  printf 'runs=%s\n' "${#CONFIGS[@]}"
  printf 'progress_log=%s\n' "$PROGRESS_LOG"
  printf 'skip_existing_final=true\n'
  printf '============================================================\n'
  printf 'RUN_ORDER\n'
  printf '%s\n' "${CONFIGS[@]}"
  printf '============================================================\n'
} >> "$PROGRESS_LOG"

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    {
      printf 'FAILED stage=%s date=%s reason=missing_config config=%s\n' "$STAGE" "$(date -Is)" "$config"
    } >> "$PROGRESS_LOG"
    exit 2
  fi

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
    if [[ "$CONTINUE_ON_ERROR" != true ]]; then
      exit "$status"
    fi
  fi
done

{
  printf '\n============================================================\n'
  printf 'ADAPTER TUNING STAGE COMPLETE stage=%s date=%s\n' "$STAGE" "$(date -Is)"
  printf '============================================================\n'
} >> "$PROGRESS_LOG"