---
license: mit
tags:
  - stable-diffusion-xl
  - sdxl
  - ip-adapter
  - product-photography
  - e-commerce
  - text-to-image
base_model: stabilityai/stable-diffusion-xl-base-1.0
library_name: diffusers
---

# StudioDiffusion IP-Adapter (Shopify / Etsy / eBay)

Three **IP-Adapter** weight sets trained on top of [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), each targeting a distinct e-commerce platform aesthetic:

- **Shopify** — clean white / neutral backgrounds, studio lighting, minimal props, high contrast subject separation.
- **Etsy** — warm color temperature, lifestyle / craft props, natural light, textured surfaces, artisanal hand-crafted feel.
- **eBay** — bright even lighting, plain or gradient background, sharp focus on subject, utilitarian clarity.

Companion code and training pipeline: **https://github.com/s-zx/StudioDiffusion**

## Repository layout

| Path | Contents |
|---|---|
| `shopify/final/{image_proj_model,ip_attn_processors}.pt` | Shopify checkpoint @ step 3000 |
| `shopify/train.log` | Shopify val-loss per 250 steps |
| `etsy/final/{image_proj_model,ip_attn_processors}.pt` | Etsy checkpoint @ step 3000 |
| `etsy/checkpoint-500/{image_proj_model,ip_attn_processors}.pt` | **Recommended** Etsy checkpoint — best val loss, before mild overfit |
| `etsy/train.log` | Etsy val-loss per 250 steps |
| `ebay/final/{image_proj_model,ip_attn_processors}.pt` | eBay checkpoint @ step 3000 |
| `ebay/train.log` | eBay val-loss per 250 steps |

Each checkpoint follows the `IPAdapterSDXL.save_pretrained` format defined in [`adapters/ip_adapter/model.py`](https://github.com/s-zx/StudioDiffusion/blob/main/adapters/ip_adapter/model.py). Two files per checkpoint: `image_proj_model.pt` (CLIP-embed → token projection) and `ip_attn_processors.pt` (injected K/V weights for every cross-attention block of the SDXL UNet).

## Usage

### Download

```python
from huggingface_hub import snapshot_download

# Full set (~5.6 GB)
snapshot_download(
    repo_id="jasonshen8848/StudioDiffusion-ip-adapter",
    local_dir="checkpoints/ip_adapter",
)

# Single platform (~1.4 GB)
snapshot_download(
    repo_id="jasonshen8848/StudioDiffusion-ip-adapter",
    local_dir="checkpoints/ip_adapter",
    allow_patterns=["shopify/final/*", "shopify/train.log"],
)
```

### Generate — minimal inference example

A complete working example is at [`inference/smoke.py`](https://github.com/s-zx/StudioDiffusion/blob/main/inference/smoke.py). Core pattern:

```python
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
from torchvision import transforms

from adapters.ip_adapter.model import IPAdapterSDXL  # from the GitHub repo

device, dtype = "mps", torch.float16  # also works on CUDA with these

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype,
    ),
    torch_dtype=dtype,
).to(device)

adapter = IPAdapterSDXL.load_pretrained(
    unet=pipe.unet,
    load_directory="checkpoints/ip_adapter/shopify/final",
    image_encoder_id="openai/clip-vit-large-patch14-336",
    num_tokens=16,
    adapter_scale=1.0,
).to(device=device, dtype=dtype)

clip_transform = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])

ref = Image.open("my_product.jpg").convert("RGB")
clip_input = clip_transform(ref).unsqueeze(0).to(device=device, dtype=dtype)
with torch.no_grad():
    cond_ip, uncond_ip = adapter.encode_image(clip_input)
ip_hidden_states = torch.cat([uncond_ip, cond_ip], dim=0)  # [uncond, cond] for CFG

image = pipe(
    prompt="a professional product photograph",
    negative_prompt="blurry, low quality, distorted, artifacts",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512, width=512,
    cross_attention_kwargs={"ip_hidden_states": ip_hidden_states},
).images[0]
image.save("out.png")
```

## Training summary

| | Shopify | Etsy | eBay |
|---|---|---|---|
| Train images | 353 | 325 | 518 |
| Val images | 88 | 81 | 129 |
| Start val loss (step 250) | 0.073747 | 0.131454 | 0.058868 |
| End val loss (step 3000) | 0.072500 | 0.132335 | 0.055920 |
| Best val loss | 0.072463 @ step 2000 | **0.131412 @ step 750** | 0.055920 @ step 3000 |
| Δ val loss | **−1.7%** ↓ | **+0.7%** ↑ (mild overfit) | **−5.0%** ↓ |
| Wall-clock | ~9 h | ~9 h | ~9 h |

**Hyperparameters** (identical across platforms):

- Base: `stabilityai/stable-diffusion-xl-base-1.0`
- VAE: `madebyollin/sdxl-vae-fp16-fix`
- Image encoder: `openai/clip-vit-large-patch14-336` (frozen)
- Optimizer: AdamW, lr=1e-4, (β₁, β₂)=(0.9, 0.999), wd=0.01
- LR schedule: cosine with 200-step warmup
- **Mixed precision: "no" (pure fp32)** — required for MPS stability
- Image size: 512×512 diffusion path; 336×336 CLIP-branch (fixed by encoder)
- Effective batch: 2 micro × 4 grad-accum = 8
- Steps: 3000 (= ~75 epochs on Shopify/Etsy, ~46 on eBay)
- Gradient checkpointing: enabled (required on 48 GB M4 Pro)
- Seed: 42

**Training data**: curated via `data/curate_platform.py` in the companion repo. Sources: Amazon Berkeley Objects (ABO), LAION-Aesthetics, DeepFashion2. ~400 images per platform selected by CLIP platform-prompt similarity + category balancing; 80/20 train/val split recorded in manifest CSVs.

**Hardware**: Apple MacBook Pro M4 Pro, 48 GB unified memory, PyTorch MPS backend.

## Known limitations

- **Captions are identity placeholders.** Training used `"a product photo"` for every sample (BLIP-2 caption generation was deferred). Text conditioning therefore provides minimal per-sample variance; all platform aesthetic signal flows through the IP-Adapter image branch.
- **Shopify adapter may over-desaturate color.** In qualitative spot checks, the Shopify adapter can push outputs towards white even when the reference product has a distinct color. If color fidelity matters, try `adapter_scale=0.5–0.75` at inference.
- **Etsy is mildly overfit after step 750.** Val loss rose ~0.7% from step 750 → 3000. The `final/` checkpoint is stylistically the strongest but diverges more from the reference content. **For content-preserving generation, prefer `etsy/checkpoint-500/`** (closest available to the val-loss optimum).
- **fp32 training was forced by MPS.** On Apple Silicon, autocast fp16/bf16 for SDXL + IP-Adapter raises an MPS `NDArrayMatrixMultiplication` assertion on the first forward pass. These weights are architecturally compatible with fp16 inference (verified on MPS — see the example above), but **fp16 / bf16 training** of this adapter configuration on CUDA has not been tested here.
- **No ControlNet / segmentation integration in these weights.** The companion repo plans a SAM2 + seg-trained ControlNet path; these checkpoints were trained without any spatial conditioning signal.

## License

MIT — matches the parent project.

Individual dataset licenses (ABO CC BY-NC 4.0, DeepFashion2 gated, LAION CC BY 4.0) apply to the *training data*, not to these weight files. Please consult those upstream licenses before commercial use.

## Citation

If you use these checkpoints, please cite the parent project:

```bibtex
@misc{studiodiffusion2026,
  title  = {StudioDiffusion: Training Platform-Specific Aesthetic Adapters for Product
            Photography Using Segmentation-Conditioned Diffusion Models},
  author = {Shen, Jason and contributors},
  year   = {2026},
  howpublished = {\url{https://github.com/s-zx/StudioDiffusion}},
  note   = {CS 7643 Deep Learning final project, Georgia Tech}
}
```
