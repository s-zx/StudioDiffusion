# IP-Adapter Training Report (2026-04-21)

**Hardware**: M4 Pro 48 GB (MPS, fp32) | **Branch**: `feat/ip-adapter-manifest-driven`

## Config

Base: SDXL 1.0 + `madebyollin/sdxl-vae-fp16-fix` | Image encoder: CLIP ViT-L/14-336
Steps: 3000 | Effective batch: 8 (bs=2 × ga=4) | LR: 1e-4 cosine | Image size: 512 | Grad ckpt: on

## Data (manifest CSVs, 80/20 split)

| Platform | Train | Val |
|---|---|---|
| Shopify | 353 | 88 |
| Etsy | 325 | 81 |
| eBay | 518 | 129 |

Captions: fallback `"a product photo"` (no BLIP-2 generation yet)

## Shopify Results — 9h wall-clock

```
 step | val_loss
------+---------
  250 | 0.07375
  500 | 0.07336
 1000 | 0.07287
 1500 | 0.07258
 2000 | 0.07246  ← plateau begins
 2500 | 0.07248
 3000 | 0.07250  ← converged
```

Total: **−1.7% MSE**. Plateau after step 2000. No overfitting. Checkpoint at `checkpoints/ip_adapter/shopify/final/`.

## Etsy / eBay

Pending. Same config, ~9h each.

## Notes

- MPS does not support mixed precision for SDXL+IP-Adapter (fp16/bf16 both hit matmul dtype assertion); fp32 only
- Training at 512 (not native 1024); CLIP input is 336² regardless — aesthetic signal unaffected
- MSE decrease confirms learned conditioning; final quality requires inference + evaluation suite
- LoRA teammates: `ProductDataset` signature changed to `(platform_dir, split, image_size)`; `adapters/lora/train.py` already updated
