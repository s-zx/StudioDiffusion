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

See `data/` for download and curation scripts.

---

## Evaluation Metrics

| Metric | Tool | Goal |
|--------|------|------|
| Platform aesthetic alignment | CLIP cosine sim + cluster accuracy | Primary |
| Product identity fidelity | DINOv2 feature similarity | Foreground preservation |
| Multi-dim aesthetic scoring | LAION Aesthetic Predictor v2 | Quality |
| Boundary preservation | Round-trip IoU + LPIPS | Mask fidelity |

---

## Setup

```bash
git clone https://github.com/<your-org>/StudioDiffusion.git
cd StudioDiffusion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Download datasets (see `data/` scripts), then:

```bash
# Train IP-Adapter for Shopify
bash scripts/train_ip_adapter.sh shopify

# Train LoRA for Etsy
bash scripts/train_lora.sh etsy

# Run full evaluation
bash scripts/run_eval.sh
```

---

## References

1. Ravi et al. (2024). *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714
2. Qin et al. (2020). *U²-Net: Going Deeper with Nested U-Structure.* PR.
3. Podell et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* arXiv:2307.01952
4. Zhang & Agrawala (2023). *Adding Conditional Control to Text-to-Image Diffusion Models.* ICCV 2023.
5. Ye et al. (2023). *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models.* arXiv:2308.06721
6. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
7. Liu et al. (2025). *ICAS: Image Conditioned Aesthetic Style Transfer.*
8. Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
9. Schuhmann et al. (2022). *LAION-5B: An open large-scale dataset for training next generation image-text models.* NeurIPS 2022.

---

## Team

| Name | Modules |
|------|---------|
| TBD | Segmentation (SAM2 + U²Net evaluation) |
| TBD | Adapter training (IP-Adapter) |
| TBD | Adapter training (LoRA) |
| TBD | Evaluation suite |
| TBD | Data curation pipeline |
