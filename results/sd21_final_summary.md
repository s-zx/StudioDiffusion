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
- Recomputed held-out image-space metrics on the leakage-free final-eval bundle using the paired
  `final eval clean val` and `final eval original inputs` packages:
  - `results/final_eval_metrics.json`
  - `results/final_eval_metrics.csv`
  - `results/sd21_ip_adapter_final_eval_metrics.csv`
- Added a second local-environment reproduction for the same held-out image bundle:
  - `results/final_eval_metrics_local_env.json`
  - `results/final_eval_metrics_local_env.csv`

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

## Held-out image-space signals (IP-Adapter only)

These metrics were recomputed on the leakage-free final-eval bundle by comparing generated outputs
 against the paired original validation inputs for each platform.

- Shopify IP-Adapter:
  - mean target CLIP similarity: `0.746`
  - k-NN platform accuracy: `0.333`
  - CLIP diversity (mean pairwise distance): `0.348`
  - FID vs. held-out original inputs: `233.802`
- Etsy IP-Adapter:
  - mean target CLIP similarity: `0.852`
  - k-NN platform accuracy: `1.000`
  - CLIP diversity (mean pairwise distance): `0.168`
  - FID vs. held-out original inputs: `273.441`
- eBay IP-Adapter:
  - mean target CLIP similarity: `0.766`
  - k-NN platform accuracy: `0.595`
  - CLIP diversity (mean pairwise distance): `0.300`
  - FID vs. held-out original inputs: `249.962`

Interpretation:

- Etsy remains the strongest platform in aesthetic alignment, matching the training-loss story that
  the adapter strongly internalized Etsy styling.
- Etsy also has the lowest diversity of the three IP-Adapter exports, which is directionally
  consistent with the mild overfitting signal from the validation-loss curve.
- Shopify and eBay remain more mixed in platform classification accuracy, which helps explain why
  the qualitative results feel less cleanly separated there than they do for Etsy.

## Environment reproducibility note

The new `final_eval_metrics_local_env.*` files show that CLIP-derived absolute values shift across
local environments, while FID is effectively unchanged.

What remained stable across both reproductions:

- Etsy stayed the strongest-aligned platform by both target CLIP similarity and k-NN accuracy.
- Shopify stayed the weakest platform in CLIP-based alignment.
- FID values were nearly identical across environments, differing only at very small floating-point
  scales.

What changed:

- Absolute CLIP similarity values moved downward in the second local environment.
- k-NN accuracy for Shopify and eBay also shifted downward in the second local environment.
- CLIP diversity values increased in the second local environment, again indicating that CLIP-based
  metrics are more environment-sensitive than FID in this setup.

Practical conclusion:

- The safest claims for `SD-21` are trend-based rather than absolute-value-based.
- The overfitting conclusion itself does **not** change: Etsy is still the only platform with a
  clear validation-loss rise after its best checkpoint, and the held-out image metrics still place
  Etsy as the strongest style-specialized adapter family.

## Remaining non-blocking note

The full `data/platform_sets/` bundle is still absent from this workspace, so these image-space
metrics were recomputed on the leakage-free final-eval bundle rather than over the original broader
 platform-set directories. That is still a valid held-out evaluation for the final report and is a
 stronger basis than the earlier loss-only summary.
