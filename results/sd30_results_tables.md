# SD-30 Results Tables and Figures

Generated from `final eval clean val/metadata/*` and local SD-21 overfitting outputs.

## Table 1. Clean split sizes

| Platform | Clean Train | Clean Val-Only | Excluded Val Overlap |
| --- | --- | --- | --- |
| shopify | 325 | 75 | 12 |
| etsy | 320 | 80 | 1 |
| ebay | 321 | 79 | 20 |

## Table 2. Final evaluation output counts and runtime

| Platform | Adapter | Outputs | Mean Seconds | Median Seconds |
| --- | --- | --- | --- | --- |
| shopify | ip_adapter | 75 | 23.671 | 23.66 |
| shopify | lora | 75 | 22.36 | 22.376 |
| etsy | ip_adapter | 80 | 23.702 | 23.677 |
| etsy | lora | 80 | 22.449 | 22.414 |
| ebay | ip_adapter | 79 | 24.055 | 24.017 |
| ebay | lora | 79 | 22.385 | 22.343 |

## Table 3. IP-Adapter overfitting summary

| Platform | Best Step | Best Val Loss | Final Step | Final Val Loss | Delta % |
| --- | --- | --- | --- | --- | --- |
| shopify | 2000 | 0.072463 | 3000 | 0.0725 | 0.051 |
| etsy | 750 | 0.131412 | 3000 | 0.132335 | 0.702 |
| ebay | 3000 | 0.05592 | 3000 | 0.05592 | 0.0 |

## Evaluation setup summary

- Total generated outputs: 468
- Generated successfully: 468
- Adapters evaluated: LoRA and IP-Adapter
- Platforms evaluated: Shopify, Etsy, eBay
- Resolution: 1024x1024
- Steps: 40
- Guidance scale: 8.5
- ControlNet model: `diffusers/controlnet-canny-sdxl-1.0`
- Adapter scale (all models in clean eval): `0.1`

## Figure assets

- `final eval clean val/galleries/overview_first_8_per_combo.jpg`
- `final eval clean val/galleries/shopify_ip_adapter_contact_sheet.jpg`
- `final eval clean val/galleries/shopify_lora_contact_sheet.jpg`
- `final eval clean val/galleries/etsy_ip_adapter_contact_sheet.jpg`
- `final eval clean val/galleries/etsy_lora_contact_sheet.jpg`
- `final eval clean val/galleries/ebay_ip_adapter_contact_sheet.jpg`
- `final eval clean val/galleries/ebay_lora_contact_sheet.jpg`

## Notes

- This folder is a leakage-free clean validation set built from `data/platform_sets_clean/*/val_only`.
- The clean-eval package contains qualitative outputs and metadata, but not raw reference bundles or metric JSONs for CLIP/FID/LPIPS.
- SD-21 local train/val overfitting analyses are incorporated here through the `results/ip_adapter_*_overfit.json` files.
