# SD-30 Report Section Draft

## Final evaluation protocol

We generated a leakage-free clean validation set using `data/platform_sets_clean/*/val_only` and evaluated both adapter families (`IP-Adapter` and `LoRA`) on all three target platforms. The final clean evaluation bundle contains 468 generated outputs across Shopify, Etsy, and eBay. All runs used the same generation settings: 1024x1024 resolution, 40 denoising steps, guidance scale 8.5, and a low ControlNet conditioning scale of 0.05 with `diffusers/controlnet-canny-sdxl-1.0`.

## Quantitative summary

| Platform | Clean Train | Clean Val-Only | Excluded Val Overlap |
| --- | --- | --- | --- |
| shopify | 325 | 75 | 12 |
| etsy | 320 | 80 | 1 |
| ebay | 321 | 79 | 20 |

| Platform | Adapter | Outputs | Mean Seconds | Median Seconds |
| --- | --- | --- | --- | --- |
| shopify | ip_adapter | 75 | 23.671 | 23.66 |
| shopify | lora | 75 | 22.36 | 22.376 |
| etsy | ip_adapter | 80 | 23.702 | 23.676 |
| etsy | lora | 80 | 22.449 | 22.39 |
| ebay | ip_adapter | 79 | 24.055 | 24.017 |
| ebay | lora | 79 | 22.385 | 22.343 |

| Platform | Best Step | Best Val Loss | Final Step | Final Val Loss | Delta % |
| --- | --- | --- | --- | --- | --- |
| shopify | 2000 | 0.072463 | 3000 | 0.0725 | 0.051 |
| etsy | 750 | 0.131412 | 3000 | 0.132335 | 0.702 |
| ebay | 3000 | 0.05592 | 3000 | 0.05592 | 0.0 |

These results show that the clean validation protocol is balanced across platform-adapter combinations, with 468 / 468 runs completing successfully. IP-Adapter runs were consistently slower than LoRA runs by roughly 1.2 to 1.7 seconds per sample in this clean-eval export. Overfitting analysis on the published IP-Adapter checkpoints indicates that Etsy is the only platform with a meaningful post-optimum validation-loss increase, while Shopify and eBay remain effectively stable through the final checkpoint.

## Qualitative figure plan

Use `fig:overview` as the main paper figure for side-by-side qualitative comparison across platforms and adapter types. Use the six per-combination contact sheets as appendix figures or backup slides. The Shopify sheets are especially useful for discussing failure cases where the model drifts toward human or mannequin-like presentations for wearable products, while the Etsy and eBay sheets better highlight scene-style and background-style differences.

## Key takeaways

1. The clean validation export is large enough to support a meaningful final qualitative comparison across all six platform-adapter combinations.
2. The runtime metadata is strong enough to justify a small throughput table in the final report.
3. The current repo already supports an honest narrative: strong clean-eval coverage, clear overfitting conclusions for IP-Adapter, and visible qualitative failure modes that can be discussed rather than hidden.
