#!/usr/bin/env bash
# Run the full evaluation suite across all platforms and adapter types.
#
# Usage:
#   bash scripts/run_eval.sh

set -euo pipefail

PLATFORMS=("shopify" "etsy" "ebay")
ADAPTERS=("ip_adapter" "lora")
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

for PLATFORM in "${PLATFORMS[@]}"; do
  for ADAPTER in "${ADAPTERS[@]}"; do
    CKPT="checkpoints/${ADAPTER}/${PLATFORM}/final"
    if [[ ! -d "$CKPT" ]]; then
      echo "[SKIP] $ADAPTER/$PLATFORM — checkpoint not found at $CKPT"
      continue
    fi

    echo "=== Evaluating $ADAPTER / $PLATFORM ==="
    python - <<PYEOF
import json
from pathlib import Path
from evaluation import CLIPPlatformAlignment, DINOv2Fidelity, AestheticScorer, BoundaryPreservation

platform  = "$PLATFORM"
adapter   = "$ADAPTER"
gen_dir   = Path("outputs") / adapter / platform
ref_dirs  = {
    "shopify": Path("data/platform_sets/shopify"),
    "etsy":    Path("data/platform_sets/etsy"),
    "ebay":    Path("data/platform_sets/ebay"),
}
out_file  = Path("$RESULTS_DIR") / f"{adapter}_{platform}_metrics.json"

gen_images = sorted(gen_dir.glob("*.png")) + sorted(gen_dir.glob("*.jpg"))

clip_eval = CLIPPlatformAlignment(device="cuda")
clip_eval.build_reference_embeddings(ref_dirs)
metrics = clip_eval.evaluate(gen_images, platform)

aesthetic = AestheticScorer(device="cuda")
scores    = aesthetic.score_batch([str(p) for p in gen_images])
metrics["mean_aesthetic_score"] = sum(scores) / len(scores) if scores else 0.0

out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to {out_file}")
PYEOF

  done
done

echo "=== Evaluation complete. Results in $RESULTS_DIR/ ==="
