# IP-Adapter trained checkpoints

Trained IP-Adapter weights for the three e-commerce platforms (Shopify / Etsy / eBay) are hosted on Hugging Face Hub — not in this git repository, because each checkpoint is ~1.4 GB.

## Location

**🤗 https://huggingface.co/jasonshen8848/StudioDiffusion-ip-adapter**

(Public repo. No HF login required to download.)

## What's there

| Path | ~Size | Notes |
|---|---|---|
| `shopify/final/` | 1.4 GB | Recommended Shopify checkpoint (step 3000) |
| `etsy/final/` | 1.4 GB | Final Etsy (step 3000) — mildly overfit |
| `etsy/checkpoint-500/` | 1.4 GB | **Recommended Etsy** — best val loss, pre-overfit |
| `ebay/final/` | 1.4 GB | Recommended eBay (step 3000) |
| `*/train.log` | tiny | Val-loss history per 250 steps |
| `README.md` | tiny | Full model card with training details + usage |

Total: ~5.6 GB.

## Download

```python
from huggingface_hub import snapshot_download

# All checkpoints
snapshot_download(
    repo_id="jasonshen8848/StudioDiffusion-ip-adapter",
    local_dir="checkpoints/ip_adapter",
)

# Just one platform
snapshot_download(
    repo_id="jasonshen8848/StudioDiffusion-ip-adapter",
    local_dir="checkpoints/ip_adapter",
    allow_patterns=["shopify/final/*"],
)
```

After download, the local layout matches what `scripts/train_ip_adapter.sh` would have produced, so existing code (`inference/smoke.py`, `evaluation/*`) finds the checkpoints at the expected paths.

## Using

See [`inference/smoke.py`](../inference/smoke.py) for a runnable minimal example. The loader is `adapters.ip_adapter.model.IPAdapterSDXL.load_pretrained(...)` — not diffusers' stock `pipe.load_ip_adapter()`, since our checkpoint layout is two files (`image_proj_model.pt` + `ip_attn_processors.pt`), not the H94 single-file format.

## Re-uploading (maintainer)

If the checkpoints get retrained or you fork the project and want to publish your own version:

1. Edit `REPO_ID` in [`share/upload_ip_adapter_to_hf.py`](./upload_ip_adapter_to_hf.py) to your HF username.
2. Edit [`share/HF_MODEL_CARD.md`](./HF_MODEL_CARD.md) with your own training details.
3. Run `hf auth login` once (cache your HF token).
4. `python share/upload_ip_adapter_to_hf.py`

The upload script is idempotent — re-runs only push changed files.

## Known gotchas (see full model card on HF for details)

- Trained on Apple MPS with **pure fp32** because autocast is broken for this stack on Apple Silicon. fp16 inference works fine; fp16 training is untested on CUDA.
- **Captions were constant fallback strings** during training — all per-platform signal flows through image conditioning.
- **Shopify adapter over-desaturates color** at `adapter_scale=1.0`. Try 0.5–0.75.
- **Etsy adapter mildly overfits after step 750** — prefer `etsy/checkpoint-500/` for content-preserving generation.
