# SD-21 Final Summary

## Scope completed

`SD-21` is complete for the published IP-Adapter checkpoints.

What was completed in-repo:

- Added reusable overfitting-analysis tooling:
  - `scripts/run_overfit_analysis.py`
  - `evaluation/clip_diversity.py`
  - `evaluation/fid.py`
- Updated LoRA training so future runs emit train/val logs suitable for the same analysis.
- Downloaded the public IP-Adapter train logs from Hugging Face and generated local analysis outputs:
  - `results/ip_adapter_shopify_overfit.json`
  - `results/ip_adapter_etsy_overfit.json`
  - `results/ip_adapter_ebay_overfit.json`

## Final conclusions

- Shopify: validation loss improved through training and stayed near its best value at step 3000.
- Etsy: mild overfitting after the best validation point at step 750.
- eBay: validation loss improved steadily through the final checkpoint.

## Exact local outputs

- Shopify:
  - best val loss `0.072463` at step `2000`
  - final val loss `0.072500`
  - delta from best to final: `+0.051%`
- Etsy:
  - best val loss `0.131412` at step `750`
  - final val loss `0.132335`
  - delta from best to final: `+0.702%`
- eBay:
  - best val loss `0.055920` at step `3000`
  - final val loss `0.055920`
  - delta from best to final: `0.000%`

## Remaining non-blocking note

The repo now has code for `CLIP diversity` and `FID`, but the local workspace still does not include `data/platform_sets/` reference image bundles, so those two metrics were not recomputed here.
