# SD-19 / SD-21 Completion Notes

## SD-19

Status: functionally complete, with real local demo runs.

Evidence:

- End-to-end generation entrypoint is implemented in `inference/generate.py`.
- Asset preparation is automated by `scripts/prepare_generation_assets.sh`.
- A real input image was processed locally:
  - input: `/Users/wanghuiling/Downloads/batch-1.png`
  - mask: `outputs/generate_smoke/shopify_batch1_mask.png`
  - control image: `outputs/generate_smoke/shopify_batch1_control.png`
  - generated output: `outputs/generate_smoke/shopify_batch1_generated.png`
- Additional tuned local demos were run to improve qualitative output:
  - `outputs/generate_smoke/shopify_batch1_generated_fixed.png`
  - `outputs/generate_smoke/shopify_batch1_generated_fixed2.png`
  - `outputs/generate_smoke/shopify_lipstick_generated.png`
- During completion work, offline loading was fixed in `adapters/ip_adapter/model.py` by making `CLIPImageProcessor` optional when cached assets are incomplete but explicit torchvision preprocessing is already used.
- During completion work, `inference/generate.py` was updated to derive a cleaner IP-Adapter reference crop from the segmentation mask and to apply heuristics for background-selected SAM2 masks.

Current limitation:

- Output quality is now strong enough for a working demo, but still benefits from prompt/category-specific tuning and from better segmentation prompts on wearable products like headbands.

## SD-21

Status: completed for the published IP-Adapter checkpoints; tooling completed for future LoRA analysis.

Evidence:

- Overfitting-analysis tooling is now implemented:
  - `scripts/run_overfit_analysis.py`
  - `evaluation/clip_diversity.py`
  - `evaluation/fid.py`
- LoRA training now emits train/val logs suitable for overfitting analysis:
  - `adapters/lora/train.py`
- Published IP-Adapter results have been captured in:
  - `results/sd21_ip_adapter_overfitting_summary.json`
- Published IP-Adapter train logs were downloaded locally and analyzed with the repo's own script:
  - `results/ip_adapter_shopify_overfit.json`
  - `results/ip_adapter_etsy_overfit.json`
  - `results/ip_adapter_ebay_overfit.json`
  - summary write-up: `results/sd21_final_summary.md`

Summary:

- Shopify: no clear overfit (`-1.7%` validation-loss delta vs. best)
- Etsy: mild overfit (`+0.7%` validation-loss delta vs. best)
- eBay: no clear overfit (`-5.0%` validation-loss delta vs. best)

Remaining limitation:

- Full local recomputation of `CLIP diversity` and `FID` was not run here because the platform reference image bundles are not present under `data/platform_sets/` in this workspace.
- LoRA-side overfitting analysis is tooling-ready, but still depends on actual LoRA train logs and generated outputs.
