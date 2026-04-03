# Data Directory

This directory contains scripts for downloading and curating all datasets used
in StudioDiffusion. **Raw data files are gitignored** — run the download scripts
to populate `data/raw/` locally.

## Directory Layout (after running scripts)

```
data/
├── raw/
│   ├── abo/                    # Amazon Berkeley Objects
│   ├── deepfashion2/           # DeepFashion2
│   └── laion_aesthetics/       # LAION-Aesthetics subset
├── platform_sets/
│   ├── shopify/                # ~200–500 curated images
│   ├── etsy/
│   └── ebay/
└── processed/
    ├── masks/                  # SAM2-generated segmentation masks
    └── captions/               # BLIP-2 generated captions
```

## Usage

```bash
# 1. Download ABO (requires AWS CLI)
bash data/download_abo.sh

# 2. Download LAION-Aesthetics subset
bash data/download_laion.sh

# 3. Curate platform-specific sets from ABO + DeepFashion2
python data/curate_platform.py --source data/raw/abo --output data/platform_sets

# 4. Generate BLIP-2 captions for training images
python data/generate_captions.py --input data/platform_sets --output data/processed/captions
```

## DeepFashion2

Access requires filling out a Google Form at:
https://github.com/switchablenorms/DeepFashion2

After approval, follow the instructions to download and place images under
`data/raw/deepfashion2/`.
