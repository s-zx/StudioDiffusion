#!/usr/bin/env bash
# DeepFashion2 dataset setup.
#
# DeepFashion2 requires filling out a Google Form to obtain download links.
# This script handles everything AFTER the zip files have been received.
#
# ============================================================
# STEP 1 — Request access (one-time, takes ~1 business day)
# ============================================================
#   1. Fill out the form at:
#      https://docs.google.com/forms/d/e/1FAIpQLSd8BsFNHm7-WHijWS-kSzQmRm2M2WbXMNJSd8hfL3E2J-eXEQ/viewform
#   2. You will receive an email with a Google Drive link to:
#        train.zip        (~7 GB)   — 191,961 shop + user images + annotations
#        validation.zip   (~1.7 GB) — 32,153 images + annotations
#      (test set has no public annotations — we use validation as our test split)
#   3. Download both zip files to:
#        data/raw/deepfashion2/zips/train.zip
#        data/raw/deepfashion2/zips/validation.zip
#
# ============================================================
# STEP 2 — Run this script to unzip and organise
# ============================================================
#   bash data/download_deepfashion2.sh
#
# ============================================================
# STEP 3 — Preprocess annotations → mask PNGs
# ============================================================
#   python data/preprocess_deepfashion2.py

set -euo pipefail

ZIPS_DIR="data/raw/deepfashion2/zips"
DEST="data/raw/deepfashion2"

# ---- Verify zip files exist ----
if [[ ! -f "$ZIPS_DIR/train.zip" || ! -f "$ZIPS_DIR/validation.zip" ]]; then
  cat <<'MSG'
[DeepFashion2] ERROR: zip files not found.

Expected locations:
  data/raw/deepfashion2/zips/train.zip
  data/raw/deepfashion2/zips/validation.zip

Please request access via the Google Form:
  https://docs.google.com/forms/d/e/1FAIpQLSd8BsFNHm7-WHijWS-kSzQmRm2M2WbXMNJSd8hfL3E2J-eXEQ/viewform

After approval, download both zip files to data/raw/deepfashion2/zips/
then re-run this script.
MSG
  exit 1
fi

# ---- Unzip train ----
echo "[DeepFashion2] Extracting train.zip (~7 GB, this may take several minutes)..."
mkdir -p "$DEST/train"
unzip -q -o "$ZIPS_DIR/train.zip" -d "$DEST/train"
echo "[DeepFashion2] train/ extracted."

# ---- Unzip validation ----
echo "[DeepFashion2] Extracting validation.zip (~1.7 GB)..."
mkdir -p "$DEST/validation"
unzip -q -o "$ZIPS_DIR/validation.zip" -d "$DEST/validation"
echo "[DeepFashion2] validation/ extracted."

# ---- Verify expected directory structure ----
# Expected after unzip:
#   data/raw/deepfashion2/train/image/       (JPEG files)
#   data/raw/deepfashion2/train/annots/      (one JSON per image)
#   data/raw/deepfashion2/validation/image/
#   data/raw/deepfashion2/validation/annots/

for SPLIT in train validation; do
  IMG_DIR="$DEST/$SPLIT/image"
  ANN_DIR="$DEST/$SPLIT/annots"
  if [[ ! -d "$IMG_DIR" ]]; then
    echo "[DeepFashion2] WARNING: expected directory not found: $IMG_DIR"
    echo "  The zip may have a different internal layout. Check and adjust paths in"
    echo "  data/preprocess_deepfashion2.py if needed."
  else
    N_IMGS=$(find "$IMG_DIR" -name "*.jpg" | wc -l | tr -d ' ')
    N_ANNS=$(find "$ANN_DIR" -name "*.json" | wc -l | tr -d ' ')
    echo "[DeepFashion2] $SPLIT: $N_IMGS images, $N_ANNS annotation files"
  fi
done

echo ""
echo "[DeepFashion2] Setup complete."
echo "Next step: python data/preprocess_deepfashion2.py"
