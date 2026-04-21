#!/usr/bin/env bash
# Build teammate data archives under share/. Run from anywhere.
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"
DATE="$(date +%Y%m%d)"
COMMIT="$(git rev-parse HEAD)"

# Refresh snapshot file with current commit
cat > share/DATA_SNAPSHOT.txt <<EOF
StudioDiffusion — data bundle snapshot
======================================

Packaged on: ${DATE}
Git commit: ${COMMIT}
Repository: https://github.com/s-zx/StudioDiffusion

Contents (FULL archive)
------------------------
- data/platform_sets/{shopify,etsy,ebay}/     curated JPEGs
- data/platform_sets/manifests/*.csv          train/val splits + category
- data/processed/abo_manifest.csv
- data/processed/deepfashion2_manifest.csv
- data/processed/deepfashion2/masks/          PNG masks (train + validation)

Curation: produced with curate_platform.py on multi-source pool (ABO + LAION + DF2);
see repo docs for latest flags. CSV image_path may be absolute paths from packager.

Excluded: data/raw/, clip_embedding_cache.pt
EOF

FULL="share/StudioDiffusion-datashare-FULL-${DATE}.tar.gz"
LITE="share/StudioDiffusion-datashare-PLATFORM-ONLY-${DATE}.tar.gz"

echo "[pack] FULL -> ${FULL}"
COPYFILE_DISABLE=1 tar -czf "${FULL}" \
  share/TEAM_DATA_BUNDLE_README.txt \
  share/DATA_SNAPSHOT.txt \
  data/platform_sets \
  data/processed/abo_manifest.csv \
  data/processed/deepfashion2_manifest.csv \
  data/processed/deepfashion2/masks

echo "[pack] PLATFORM-ONLY -> ${LITE}"
COPYFILE_DISABLE=1 tar -czf "${LITE}" \
  share/TEAM_DATA_BUNDLE_README.txt \
  share/DATA_SNAPSHOT.txt \
  data/platform_sets

ls -lh "${FULL}" "${LITE}"
echo "[pack] Done. Upload these files + point teammates to share/TEAM_DATA_BUNDLE_README.txt inside the archive."
