#!/bin/bash
# PACE-ICE SLURM wrapper for LoRA training.
#
# Usage:
#   sbatch scripts/sbatch_train_lora.sh shopify
#   sbatch scripts/sbatch_train_lora.sh etsy
#   sbatch scripts/sbatch_train_lora.sh ebay
#
# Adjust partition / QoS / account to match the cluster account in use.

#SBATCH --job-name=lora-train
#SBATCH --partition=ice-gpu                # or coc-gpu, depending on access
#SBATCH --gres=gpu:V100:1                  # SDXL needs >=16 GB VRAM; V100 32GB or A100
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PLATFORM="${1:?usage: sbatch scripts/sbatch_train_lora.sh <platform>}"

mkdir -p logs

# --- Environment ---
module load python/3.10 cuda/12.1
source .venv/bin/activate

# Cache HuggingFace weights on scratch to avoid blowing the home quota
# (UNet alone is ~5 GB; SDXL pipeline ~7 GB total).
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-studio-diffusion}"
# WANDB_API_KEY should be set in the user environment, NOT committed here.

echo "[sbatch] PLATFORM=$PLATFORM  HF_HOME=$HF_HOME  HOST=$(hostname)"
echo "[sbatch] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

bash scripts/train_lora.sh "$PLATFORM"
