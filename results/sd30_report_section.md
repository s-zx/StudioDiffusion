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
| ebay | lora | 79 | 22.399 | 22.359 |

| Platform | Best Step | Best Val Loss | Final Step | Final Val Loss | Delta % |
| --- | --- | --- | --- | --- | --- |
| shopify | 2000 | 0.072463 | 3000 | 0.0725 | 0.051 |
| etsy | 750 | 0.131412 | 3000 | 0.132335 | 0.702 |
| ebay | 3000 | 0.05592 | 3000 | 0.05592 | 0.0 |

| Platform | Adapter | kNN Acc. | Mean CLIP Sim | CLIP Div. | FID |
| --- | --- | --- | --- | --- | --- |
| shopify | ip_adapter | 0.333 | 0.746 | 0.348 | 233.802 |
| shopify | lora | 0.293 | 0.74 | 0.358 | 243.953 |
| etsy | ip_adapter | 1.0 | 0.852 | 0.168 | 273.441 |
| etsy | lora | 1.0 | 0.848 | 0.173 | 264.222 |
| ebay | ip_adapter | 0.595 | 0.766 | 0.3 | 249.962 |
| ebay | lora | 0.684 | 0.759 | 0.304 | 251.594 |

These results show that the clean validation protocol is balanced across platform-adapter combinations, with 468 / 468 runs completing successfully. IP-Adapter runs were consistently slower than LoRA runs by roughly 1.2 to 1.7 seconds per sample in this clean-eval export. Overfitting analysis on the published IP-Adapter checkpoints indicates that Etsy is the only platform with a meaningful post-optimum validation-loss increase, while Shopify and eBay remain effectively stable through the final checkpoint.

The refreshed clean-eval package also upgrades the eBay LoRA slice to `ebay_lora_lr2e-4_s3000` (`checkpoints/lora/ebay_lr2e-4_s3000/final`), whose training summary reports a final validation loss of `0.056886`. This means the current qualitative eBay LoRA figures are tied to the best-confirmed LoRA setting rather than the older baseline export.

Image-space metrics on the final-eval bundle add a useful second view. Both Etsy adapters achieve perfect k-NN platform classification and the highest mean CLIP similarity to their target reference set, but they also show the lowest CLIP diversity scores, which is directionally consistent with the mild Etsy overfitting signal from training loss. Shopify remains the weakest platform in platform-alignment terms, while the refreshed eBay LoRA export improves k-NN accuracy over eBay IP-Adapter and remains competitive on diversity and FID.


## Qualitative figure plan

Use `fig:overview` as the main paper figure for side-by-side qualitative comparison across platforms and adapter types. Use the six per-combination contact sheets as appendix figures or backup slides. The Shopify sheets are especially useful for discussing failure cases where the model drifts toward human or mannequin-like presentations for wearable products, while the Etsy and eBay sheets better highlight scene-style and background-style differences.

## Key takeaways

1. The clean validation export is large enough to support a meaningful final qualitative comparison across all six platform-adapter combinations.
2. The runtime metadata is strong enough to justify a small throughput table in the final report.
3. The current repo already supports an honest narrative: strong clean-eval coverage, clear overfitting conclusions for IP-Adapter, and visible qualitative failure modes that can be discussed rather than hidden.
