# StudioDiffusion

**Training Platform-Specific Aesthetic Adapters for Product Photography Using Segmentation-Conditioned Diffusion Models**

> CS 7643 Deep Learning — Final Project

---

## Overview

E-commerce sellers on Shopify, Etsy, and eBay gain organic exposure when their product images match each platform's aesthetic norms. This project investigates whether lightweight adapter modules fine-tuned on domain-specific product photography can produce measurably better platform-targeted image generation than off-the-shelf baselines.

**Key insight:** Shopify favors clean white backgrounds, Etsy prefers warm lifestyle aesthetics, and eBay prioritizes subject clarity — yet no existing generative tool targets these distributions. We close that gap.

---

## Pipeline

```
Product Image
     │
     ▼
[SAM2 Segmentation] ──── mask ────▶ [ControlNet on SDXL]
                                            │
[Platform Reference Images]                 │
     │                                      │
     ▼                                      ▼
[IP-Adapter / LoRA Training] ──────▶ [Generated Image]
                                            │
                                            ▼
                                    [Evaluation Suite]
```

### Components

| Component | Model | Trainable? | Role |
|-----------|-------|-----------|------|
| Segmentation | SAM2 | ❌ frozen | Extract foreground mask |
| Seg baseline | U²Net | ❌ frozen | Mask quality comparison |
| Spatial conditioning | ControlNet (SDXL) | ❌ frozen | Inject mask as structural signal |
| Aesthetic adapter | IP-Adapter (SDXL) | ✅ **trained** | Per-platform CLIP image conditioning |
| Aesthetic adapter | LoRA (SDXL UNet) | ✅ **trained** | Per-platform low-rank fine-tuning |

---

## Repository Structure

```
StudioDiffusion/
├── configs/                    # YAML configs for every training run
│   ├── base.yaml
│   ├── ip_adapter/             # Per-platform IP-Adapter configs
│   └── lora/                   # Per-platform LoRA configs
├── docs/                       # Data setup guide, handoff, work summary
├── share/                      # Teammate bundle README + pack_data_bundle.sh (see share/README.md)
├── data/                       # Dataset download & curation scripts
│   ├── download_abo.sh
│   ├── download_deepfashion2.sh
│   ├── download_laion.sh
│   ├── curate_platform.py      # CLIP-guided platform set curation
│   └── generate_captions.py    # BLIP-2 caption generation
├── segmentation/               # SAM2 extraction + U²Net baseline
│   ├── sam2_extractor.py
│   ├── u2net_baseline.py
│   └── evaluate_masks.py       # IoU / mask quality metrics
├── adapters/                   # Core training contribution
│   ├── ip_adapter/
│   │   ├── model.py
│   │   └── train.py
│   └── lora/
│       ├── model.py
│       └── train.py
├── evaluation/                 # All evaluation metrics
│   ├── clip_alignment.py       # Platform cluster cosine sim + accuracy
│   ├── dinov2_fidelity.py      # Product identity preservation
│   ├── aesthetic_scoring.py    # LAION Aesthetic Predictor v2
│   └── boundary_preservation.py# Round-trip IoU + LPIPS
├── inference/
│   └── generate.py             # End-to-end generation script
├── scripts/                    # Slurm / shell training launchers
│   ├── train_ip_adapter.sh
│   ├── train_lora.sh
│   └── run_eval.sh
├── notebooks/                  # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_eval.ipynb
│   └── 03_results_analysis.ipynb
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Datasets

| Dataset | Size | License | Use |
|---------|------|---------|-----|
| [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | 398K images | CC BY-NC 4.0 | Product images + seg masks |
| [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) | 491K images | Gated | Seg eval ground truth |
| [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) | Filtered LAION-5B | CC BY 4.0 | Aesthetic benchmark |
| Platform-curated sets | 200–500 imgs / platform | Derived | Adapter training data |

**Full data setup (venv, ABO, DeepFashion2, LAION, platform curation, disk notes, “share data vs run scripts”):** see **[`docs/data-setup.md`](docs/data-setup.md)**.  
Operational detail: [`docs/team-data-pipeline-handoff.md`](docs/team-data-pipeline-handoff.md). Short overview: [`docs/data-work-summary.md`](docs/data-work-summary.md).

---

## Evaluation Metrics

| Metric | Tool | Goal |
|--------|------|------|
| Platform aesthetic alignment | CLIP cosine sim + cluster accuracy | Primary |
| Diversity / mode collapse | CLIP pairwise distance + NN similarity | Overfitting detection |
| Distribution realism | FID | Overfitting / realism tradeoff |
| Product identity fidelity | DINOv2 feature similarity | Foreground preservation |
| Multi-dim aesthetic scoring | LAION Aesthetic Predictor v2 | Quality |
| Boundary preservation | Round-trip IoU + LPIPS | Mask fidelity |

---

## Setup

```bash
git clone https://github.com/s-zx/StudioDiffusion.git
cd StudioDiffusion
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"   # optional
```

**Prepare data** before training: follow **[`docs/data-setup.md`](docs/data-setup.md)** end-to-end (ABO → DeepFashion2 → LAION subset → `curate_platform.py`). Then:

```bash
# Train IP-Adapter for Shopify
bash scripts/train_ip_adapter.sh shopify

# Train LoRA for Etsy
bash scripts/train_lora.sh etsy

# Run full evaluation
bash scripts/run_eval.sh

# Summarize overfitting from a training log (+ optional image metrics)
.venv/bin/python scripts/run_overfit_analysis.py \
  --log checkpoints/ip_adapter/shopify/train.log \
  --generated-dir outputs/ip_adapter/shopify \
  --reference-dir data/platform_sets/shopify \
  --output results/ip_adapter_shopify_overfit.json
```

## Generation

The end-to-end generation entrypoint is [`inference/generate.py`](inference/generate.py).
It stitches together:

1. `SAM2` foreground extraction
2. `SDXL` image generation
3. a trained platform adapter (`IP-Adapter` or `LoRA`)

For the most stable demo path, run the bundled smoke script:

```bash
bash scripts/prepare_generation_assets.sh shopify
bash scripts/run_generate_smoke.sh path/to/product.jpg
```

This uses:

- the `shopify` IP-Adapter checkpoint
- `SAM2` foreground extraction
- the stable **no-ControlNet** path
- `512x512` generation for quick validation

Outputs are written to:

```text
outputs/generate_smoke/shopify/
├── mask.png
├── control.png
└── generated.png
```

You can also call the entrypoint directly:

```bash
.venv/bin/python inference/generate.py \
  --product path/to/product.jpg \
  --platform shopify \
  --adapter ip_adapter \
  --adapter_ckpt checkpoints/ip_adapter/shopify/final \
  --disable-controlnet \
  --device mps \
  --segmentation-device cpu \
  --dtype fp16 \
  --steps 8 \
  --height 512 \
  --width 512 \
  --mask-output outputs/shopify_mask.png \
  --control-output outputs/shopify_control.png \
  --output outputs/shopify_generated.png
```

### Current status

- `IP-Adapter` end-to-end generation is wired up and smoke-tested.
- `LoRA` is supported by the CLI entrypoint, but still depends on having trained LoRA checkpoints available locally.
- The experimental `ControlNet` branch is optional; if the ControlNet model cannot be loaded, `generate.py` can fall back to base `SDXL` unless `--strict-controlnet` is set.
- Add `--local-files-only` once your Hugging Face assets are already cached and you want fully offline generation.

### Asset preparation

If generation assets are missing, cache them with:

```bash
bash scripts/prepare_generation_assets.sh shopify
```

This prepares the minimum asset bundle for the current SD-19 demo path:

- `sdxl-vae-fp16-fix`
- `stable-diffusion-xl-base-1.0`
- `clip-vit-large-patch14-336`
- `shopify` IP-Adapter checkpoint
- `sam2_hiera_large.pt`
