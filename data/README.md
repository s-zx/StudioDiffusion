# Data Directory

This directory contains all scripts for downloading and preprocessing the
datasets used in StudioDiffusion. **Raw data files are gitignored** — run
the scripts below to populate `data/raw/` locally.

## Directory Layout (after running all scripts)

```
data/
├── raw/
│   ├── abo/
│   │   ├── metadata/           # gzipped JSON-lines listing metadata
│   │   ├── images/small/       # catalog photos (JPEG, 512px long-edge)
│   │   └── renders/
│   │       ├── png/            # 3-D render PNGs (spin-0 only)
│   │       └── segmentation_masks/  # matching binary mask PNGs
│   ├── deepfashion2/
│   │   ├── train/image/        # 191K JPEG images
│   │   ├── train/annots/       # one JSON annotation per image
│   │   ├── validation/image/
│   │   └── validation/annots/
│   └── laion_aesthetics/
│       ├── parquet/            # HuggingFace metadata parquet files
│       └── images/             # downloaded image shards (webdataset .tar)
├── processed/
│   ├── abo_manifest.csv                 # item_id, image_path, product_type, …
│   ├── deepfashion2_manifest.csv        # image_path, mask_path, category_name, split
│   ├── deepfashion2/masks/              # binary mask PNGs from polygon annotations
│   │   ├── train/
│   │   └── validation/
│   ├── captions/               # BLIP-2 generated captions (one .txt per image)
│   └── clip_embedding_cache.pt # CLIP embedding cache (speeds up re-runs)
└── platform_sets/
    ├── shopify/                # ~400 curated images
    ├── etsy/
    ├── ebay/
    └── manifests/
        ├── shopify_train.csv   # 80% split
        ├── shopify_val.csv     # 20% split
        ├── etsy_train.csv
        ├── etsy_val.csv
        ├── ebay_train.csv
        └── ebay_val.csv
```

## Full Pipeline (run in order)

### SD-8: ABO Dataset

```bash
# Download catalog images + rendered images with segmentation masks
bash data/download_abo.sh

# Parse metadata → data/processed/abo_manifest.csv
python data/preprocess_abo.py
```

`download_abo.sh` flags:
- `--images-only` — skip renders (saves ~15 GB)
- `--renders-only` — skip catalog images

### SD-6: DeepFashion2 Dataset

DeepFashion2 requires filling out a **gated Google Form** first. Run the
script to see detailed instructions, then come back once you have the zips:

```bash
bash data/download_deepfashion2.sh   # prints access instructions + unzips

# Convert polygon annotations → binary PNG masks + manifest CSV
python data/preprocess_deepfashion2.py
```

### SD-11: LAION-Aesthetics Subset

```bash
# Default: ~5K images (5 shards). LAION is supplementary — use sparingly.
bash data/download_laion.sh

# For a larger subset:
bash data/download_laion.sh --n_shards 50   # ~50K images
```

### SD-16: Platform Set Curation

```bash
# Basic (ABO only)
python data/curate_platform.py \
    --sources data/raw/abo/images/small

# Multi-source with manifest metadata (recommended)
python data/curate_platform.py \
    --sources data/raw/abo/images/small data/raw/deepfashion2/train/image \
    --abo_manifest data/processed/abo_manifest.csv \
    --df2_manifest data/processed/deepfashion2_manifest.csv \
    --n_per_platform 400 \
    --device cuda
```

CLIP embeddings are cached to `data/processed/clip_embedding_cache.pt` so
re-running with different `--n_per_platform` values is fast.

### Caption Generation (adapter training prerequisite)

```bash
python data/generate_captions.py \
    --input  data/platform_sets \
    --output data/processed/captions \
    --device cuda
```

## Dataset Licenses

| Dataset | License | Notes |
|---------|---------|-------|
| ABO | CC BY-NC 4.0 | No commercial use |
| DeepFashion2 | Restricted (research only) | Requires Google Form approval |
| LAION-Aesthetics | CC BY 4.0 | URLs only; individual image licenses vary |
