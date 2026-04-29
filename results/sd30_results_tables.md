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
| etsy | ip_adapter | 80 | 23.702 | 23.676 |
| etsy | lora | 80 | 22.449 | 22.39 |
| ebay | ip_adapter | 79 | 24.055 | 24.017 |
| ebay | lora | 79 | 22.399 | 22.359 |

## Table 3. IP-Adapter overfitting summary

| Platform | Best Step | Best Val Loss | Final Step | Final Val Loss | Delta % |
| --- | --- | --- | --- | --- | --- |
| shopify | 2000 | 0.072463 | 3000 | 0.0725 | 0.051 |
| etsy | 750 | 0.131412 | 3000 | 0.132335 | 0.702 |
| ebay | 3000 | 0.05592 | 3000 | 0.05592 | 0.0 |

## Table 4. Category coverage snapshot (unique clean-val products)

| Platform | Category | Count |
| --- | --- | --- |
| shopify | uncategorized | 5 |
| shopify | COMPUTER | 3 |
| shopify | BUILDING_MATERIAL | 2 |
| shopify | PORTABLE_ELECTRONIC_DEVICE_STAND | 2 |
| shopify | MEAT | 2 |
| shopify | SLEEPING_BAG | 1 |
| shopify | SHIPPING_BOX | 1 |
| shopify | SUNSCREEN | 1 |
| shopify | STAPLER | 1 |
| shopify | SECURITY_ELECTRONICS | 1 |
| shopify | SUITCASE | 1 |
| shopify | STORAGE_BINDER | 1 |
| shopify | CAMERA_TRIPOD | 1 |
| shopify | SAUTE_FRY_PAN | 1 |
| shopify | SKIN_CLEANING_AGENT | 1 |
| etsy | uncategorized | 73 |
| etsy | BOTTLE_RACK | 1 |
| etsy | STORAGE_BOX | 1 |

## Table 5. Final-eval quality metrics

| Platform | Adapter | kNN Acc. | Mean CLIP Sim | CLIP Div. | FID |
| --- | --- | --- | --- | --- | --- |
| shopify | ip_adapter | 0.333 | 0.746 | 0.348 | 233.802 |
| shopify | lora | 0.293 | 0.74 | 0.358 | 243.953 |
| etsy | ip_adapter | 1.0 | 0.852 | 0.168 | 273.441 |
| etsy | lora | 1.0 | 0.848 | 0.173 | 264.222 |
| ebay | ip_adapter | 0.595 | 0.766 | 0.3 | 249.962 |
| ebay | lora | 0.684 | 0.759 | 0.304 | 251.594 |

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
- eBay LoRA refresh: `ebay_lora_lr2e-4_s3000` from `checkpoints/lora/ebay_lr2e-4_s3000/final` with final val loss `0.056886`


## Figure assets

- `fig:overview`: `final eval clean val/galleries/overview_first_8_per_combo.jpg`
  - Caption: Overview of the first eight clean validation outputs for each platform-adapter combination.
  - Purpose: High-level qualitative comparison across platforms and adapter types.
- `fig:shopify_ip`: `final eval clean val/galleries/shopify_ip_adapter_contact_sheet.jpg`
  - Caption: Shopify IP-Adapter contact sheet over clean validation products.
  - Purpose: Qualitative review of clean-background studio behavior and common failure cases.
- `fig:shopify_lora`: `final eval clean val/galleries/shopify_lora_contact_sheet.jpg`
  - Caption: Shopify LoRA contact sheet over clean validation products.
  - Purpose: Qualitative review of LoRA adaptation behavior for Shopify-style outputs.
- `fig:etsy_ip`: `final eval clean val/galleries/etsy_ip_adapter_contact_sheet.jpg`
  - Caption: Etsy IP-Adapter contact sheet over clean validation products.
  - Purpose: Qualitative review of warm lifestyle styling on held-out Etsy-like products.
- `fig:etsy_lora`: `final eval clean val/galleries/etsy_lora_contact_sheet.jpg`
  - Caption: Etsy LoRA contact sheet over clean validation products.
  - Purpose: Compare LoRA styling strength and content preservation for Etsy outputs.
- `fig:ebay_ip`: `final eval clean val/galleries/ebay_ip_adapter_contact_sheet.jpg`
  - Caption: eBay IP-Adapter contact sheet over clean validation products.
  - Purpose: Qualitative review of utilitarian clarity and plain-background behavior.
- `fig:ebay_lora`: `final eval clean val/galleries/ebay_lora_contact_sheet.jpg`
  - Caption: eBay LoRA contact sheet over clean validation products.
  - Purpose: Compare LoRA adaptation behavior for eBay-style product presentation.

## Notes

- This folder is a leakage-free clean validation set built from `data/platform_sets_clean/*/val_only`.
- Category counts are deduplicated by clean validation case so the same product is not double-counted across LoRA and IP-Adapter outputs.
- The updated package refreshes the eBay LoRA slice to the best-confirmed `lr=2e-4, step=3000` checkpoint when `metadata/ebay_lora_lr2e-4_s3000_training_summary.json` is present.
- The clean-eval package now supports image-space metrics using the paired `final eval original inputs` bundle; `results/final_eval_metrics.csv` summarizes CLIP alignment, diversity, and FID for all six platform-adapter combinations.
- SD-21 local train/val overfitting analyses are incorporated here through the `results/ip_adapter_*_overfit.json` files.
- The qualitative contact sheets include both strong examples and visible failure modes; this is useful for an honest final report discussion section.
