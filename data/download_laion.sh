#!/usr/bin/env bash
# Download LAION-Aesthetics images (score >= 6.0) from limingcv/LAION_Aesthetics_512
# on HuggingFace. Each parquet shard contains ~925 images embedded as bytes —
# no external URL fetching or img2dataset needed.
#
# Requires: pip install huggingface_hub pandas Pillow
#
# Usage:
#   bash data/download_laion.sh               # default: 6 shards ≈ 5,500 images
#   bash data/download_laion.sh --n_shards 20 # ≈ 18,500 images
#   bash data/download_laion.sh --full        # all 3,041 shards ≈ 2.8M images (large!)

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
N_SHARDS=6
FULL=false

for arg in "$@"; do
  case "$arg" in
    --full)          FULL=true ;;
    --n_shards=*)    N_SHARDS="${arg#*=}" ;;
    --n_shards)      shift; N_SHARDS="$1" ;;
  esac
done

if [[ "$FULL" == "true" ]]; then
  N_SHARDS=3041
fi

echo "[LAION] Downloading ${N_SHARDS} shards from limingcv/LAION_Aesthetics_512 ..."

"$PROJ/.venv/bin/python" - <<EOF
import io, os
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image

OUT_DIR = Path("$PROJ/data/raw/laion_aesthetics/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = $N_SHARDS
total = 0
for shard_idx in range(N):
    fname = f"data/train-{shard_idx:05d}-of-03041.parquet"
    print(f"[LAION] Shard {shard_idx+1}/{N}: {fname}")
    local = hf_hub_download(
        repo_id="limingcv/LAION_Aesthetics_512",
        repo_type="dataset",
        filename=fname,
    )
    df = pd.read_parquet(local)
    shard_dir = OUT_DIR / f"shard_{shard_idx:05d}"
    shard_dir.mkdir(exist_ok=True)
    saved = 0
    for i, row in df.iterrows():
        try:
            raw = row["image"]
            if isinstance(raw, dict):
                raw = raw.get("bytes", b"")
            if not raw:
                continue
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.save(shard_dir / f"{i:06d}.jpg", quality=90)
            saved += 1
        except Exception:
            continue
    total += saved
    print(f"  Saved {saved} images")

print(f"\n[LAION] Total: {total:,} images saved to {OUT_DIR}")
EOF
