#!/usr/bin/env bash
# Cache the assets needed for SD-19 generation.
#
# Usage:
#   bash scripts/prepare_generation_assets.sh
#   bash scripts/prepare_generation_assets.sh shopify
#
# Default platform is shopify because the current smoke path is based on the
# trained Shopify IP-Adapter checkpoint.

set -euo pipefail

PLATFORM="${1:-shopify}"
PYTHON_BIN=".venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv Python at $PYTHON_BIN"
  exit 1
fi

mkdir -p checkpoints/ip_adapter/${PLATFORM}/final checkpoints

echo "=== Preparing generation assets ==="
echo "platform: $PLATFORM"

"$PYTHON_BIN" - <<PY
from pathlib import Path
import shutil
import urllib.request

from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from transformers import CLIPVisionModelWithProjection

platform = "${PLATFORM}"
repo = "jasonshen8848/StudioDiffusion-ip-adapter"
out = Path(f"checkpoints/ip_adapter/{platform}/final")
out.mkdir(parents=True, exist_ok=True)

print("[1/5] Caching VAE…")
AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

print("[2/5] Caching SDXL base…")
StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    add_watermarker=False,
)

print("[3/5] Caching CLIP vision encoder…")
CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336")

print("[4/5] Downloading IP-Adapter checkpoint…")
for filename in [
    f"{platform}/final/image_proj_model.pt",
    f"{platform}/final/ip_attn_processors.pt",
]:
    cached = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy2(cached, out / Path(filename).name)

print("[5/5] Downloading SAM2 checkpoint…")
sam2_ckpt = Path("checkpoints/sam2_hiera_large.pt")
urllib.request.urlretrieve(
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    sam2_ckpt,
)

print("Assets ready.")
PY

echo "=== Done ==="
