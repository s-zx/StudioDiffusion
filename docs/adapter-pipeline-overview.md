# Adapter Training Pipeline — Flow Diagrams & Notes

> Reference doc for `IP-Adapter` and `LoRA` training pipelines on SDXL.
> Companion to `docs/lora-implementation-plan.md`.

---

## Shared pipeline (both adapters)

```
                           ┌──────────────────────────────┐
                           │   PlatformDataset            │
                           │   (e.g. data/platform_sets/  │
                           │    etsy/*.jpg + captions)    │
                           └──────────────┬───────────────┘
                                          │  batch = {pixel_values, prompt}
                                          ▼
       ┌──────────────────────────────────────────────────────────────────┐
       │                       PREPROCESSING                              │
       │                                                                  │
       │   pixel_values (B,3,1024,1024)         prompt (str list)         │
       │           │                                  │                   │
       │           ▼                                  ▼                   │
       │     ┌──────────┐                   ┌─────────────────┐           │
       │     │   VAE    │ FROZEN            │ Text encoder 1  │ FROZEN    │
       │     │ encoder  │                   │ Text encoder 2  │ FROZEN    │
       │     │  (fp16)  │                   │   (CLIP-L+G)    │           │
       │     └────┬─────┘                   └────────┬────────┘           │
       │          │ z₀  (B,4,128,128)                │ c_text             │
       │          │                                  │ (B,77,2048)        │
       │          │                                  │ + pooled (B,1280)  │
       │          ▼                                  │                    │
       │   ┌─────────────┐                           │                    │
       │   │ Add noise   │ ε ~ N(0,I)                │                    │
       │   │ z_t = √ᾱ·z₀ │ t ~ U{0..999}             │                    │
       │   │     + √(1-ᾱ)│                           │                    │
       │   │     · ε     │                           │                    │
       │   └──────┬──────┘                           │                    │
       │          │ z_t  (B,4,128,128)               │                    │
       └──────────┼──────────────────────────────────┼────────────────────┘
                  │                                  │
                  └──────────────┬───────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────┐
                │         SDXL UNet  (ε_θ)            │
                │      ▒▒▒ FROZEN backbone ▒▒▒        │
                │   ╔═══════════════════════════════╗ │
                │   ║   ADAPTER-SPECIFIC INSERTS    ║ │  ← only these train
                │   ╚═══════════════════════════════╝ │
                └─────────────────┬───────────────────┘
                                  │ noise_pred  (B,4,128,128)
                                  ▼
                       ┌──────────────────┐
                       │  MSE(ε, ε_θ(...))│ ← scalar loss
                       └────────┬─────────┘
                                │ .backward()
                                ▼
                  ┌──────────────────────────┐
                  │ AdamW.step()             │
                  │ updates only requires_   │
                  │   grad=True params       │
                  └──────────────────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │ if step % N == 0: save   │
                  │ if step >= max_steps:    │
                  │     break                │
                  └──────────────────────────┘
```

The whole shared loop is identical for IP-Adapter and LoRA. The only difference is what's inside the **`ADAPTER-SPECIFIC INSERTS`** box.

---

## Inside the UNet — IP-Adapter

```
SDXL UNet attention block (cross-attn = attn2)
─────────────────────────────────────────────────────────────────

      latent features (Q-source)        text embedding c_text (K/V-source)
              │                                       │
              ▼                                       ▼
        ┌──────────┐                            ┌──────────┐
        │  to_q    │ FROZEN                     │  to_k    │ FROZEN
        └────┬─────┘                            └────┬─────┘
             │                                       │
             │                                       │ K_text
             │                                       ▼
             │                              ┌────────────────┐
             │                              │ softmax(Q·K_t) │ ── × V_text
             │                              └────────────────┘
             │                                       │
             │                                       │ + ────────────┐
             │                                       │               │
             │                                       │      ┌────────┴────────┐
             │                                       │      │  EXTRA branch   │
             │                                       │      │  (IP-Adapter)   │
             │                                       │      └────────┬────────┘
             │                                       │               │
             │  CLIP-image embedding                                  │
             │  c_image (from reference product image)                │
             │           │                                            │
             │           ▼                                            │
             │   ┌───────────────┐                                    │
             │   │ Image Proj    │  ◄── TRAINABLE (small MLP)         │
             │   │ (~22M params) │                                    │
             │   └───────┬───────┘                                    │
             │           │                                            │
             │           ▼                                            │
             │   ┌───────────────┐    ┌───────────────┐               │
             │   │ add_k_proj    │    │ add_v_proj    │ ◄── TRAINABLE │
             │   └───────┬───────┘    └───────┬───────┘               │
             │           │ K_img              │ V_img                 │
             │           ▼                    ▼                       │
             │     ┌────────────────────────────┐                     │
             └────►│ softmax(Q·K_img) × V_img   │─────────────────────┘
                   └────────────────────────────┘
                                │
                                ▼
                          ┌──────────┐
                          │  to_out  │ FROZEN
                          └────┬─────┘
                               ▼
                          (next block)

TRAINABLE: ~22M params  (image_proj MLP + add_k_proj + add_v_proj per attn layer)
FROZEN:    ~2.6B params (everything else in the UNet)
```

**Key insight:** IP-Adapter **adds new computation paths** (the right side of the diagram). The original cross-attn with text is left intact. At inference, you blend the two with a `scale` knob (`scale=0` → pure SDXL, `scale=1` → full image conditioning).

---

## Inside the UNet — LoRA

```
SDXL UNet attention block (any attn — self OR cross)
─────────────────────────────────────────────────────────────────

      input features x  (B, seq, d_in)
              │
     ┌────────┴────────┐
     │                 │
     │ FROZEN base     │ TRAINABLE delta (LoRALinear wrapper)
     │                 │
     ▼                 ▼
  ┌──────┐      ┌─────────────┐    ┌─────────────┐
  │ to_q │      │   lora_A    │ ─► │   lora_B    │
  │ (W₀) │      │ (rank=16,   │    │ (d_out, 16) │
  │      │      │  d_in)      │    │ init = 0    │
  └──┬───┘      └──────┬──────┘    └──────┬──────┘
     │                 │                  │
     │ W₀·x            └─────► A·x ──────►│ B·A·x
     │                                    │
     │                          × (alpha/rank)
     │                                    │
     └──────────────► + ◄─────────────────┘
                      │
                      ▼
                y = W₀·x + (α/r)·B·A·x

Same wrapping is applied to:  to_q, to_k, to_v, to_out.0,
                              add_q_proj, add_k_proj, add_v_proj, to_add_out
… in EVERY attention block of the UNet (~140 wrapped Linears).

TRAINABLE:  ~5M params  (rank=16 × d_in + d_out × rank, summed over all wrapped layers)
            ~0.2% of UNet
FROZEN:     ~2.6B params
```

**Key insight:** LoRA **modifies existing computation paths**. There's no new branch — every wrapped Linear gets a low-rank "side car" added to its output. At inference, you can either:
- Keep the wrapper (slow path: `base(x) + scaling·B·A·x` = 2 matmuls), or
- **Merge** the delta into the base weights once (`W ← W₀ + scaling·B·A`) → indistinguishable from a normal SDXL UNet at runtime, zero overhead.

---

## Side-by-side: where the loss lands

```
                          backward() flows everywhere ↘
                                                       │
                                                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                         IP-Adapter                           │
  │                                                              │
  │  loss = MSE(ε, UNet+IPAdapter(z_t, t, c_text, c_image))      │
  │                                       └──────────────┬───┘   │
  │  grads land in:                                      │       │
  │    image_proj.{0,2}.{weight,bias}    ◄───────────────┤       │
  │    add_k_proj.weight  (per attn2)    ◄───────────────┤       │
  │    add_v_proj.weight  (per attn2)    ◄───────────────┘       │
  │  grads ignored for everything else                           │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │                            LoRA                              │
  │                                                              │
  │  loss = MSE(ε, UNet_with_LoRA(z_t, t, c_text))               │
  │                                  └──────────────┬───┘        │
  │  grads land in:                                 │            │
  │    every wrapped LoRALinear.lora_A   ◄──────────┤            │
  │    every wrapped LoRALinear.lora_B   ◄──────────┘            │
  │  grads ignored for everything else                           │
  └──────────────────────────────────────────────────────────────┘
```

---

## At inference (`inference/generate.py`)

```
text prompt + product photo + SAM2 mask
        │              │            │
        ▼              ▼            ▼
   text encoder   CLIP image    ControlNet(SDXL)
        │           encoder         │
        │              │            │ spatial conditioning
        │              │            │
        ▼              ▼            ▼
        ╔═══════════════════════════════════════════════╗
        ║   30-step DDIM denoise loop (z_T → z_0)       ║
        ║                                               ║
        ║   for t in [999, 966, ..., 0]:                ║
        ║       noise_pred = UNet(                      ║
        ║           z_t, t,                             ║
        ║           c_text,                             ║
        ║           IP_Adapter(c_image)  ── if loaded   ║
        ║           + LoRA deltas        ── if loaded   ║
        ║           + ControlNet(mask)   ── always      ║
        ║       )                                       ║
        ║       z_{t-1} = scheduler.step(noise_pred, t) ║
        ╚═══════════════════════════════════════════════╝
                            │ z_0  (B,4,128,128)
                            ▼
                       ┌─────────┐
                       │   VAE   │
                       │ decoder │
                       └────┬────┘
                            │
                            ▼
                  generated 1024×1024 image
```

You can stack IP-Adapter and LoRA at inference — they don't conflict because they touch different params (IP-Adapter adds new K/V; LoRA modifies existing Q/K/V/out).

---

## Summary table

| Aspect | IP-Adapter | LoRA |
|---|---|---|
| **Trained params** | ~22M (image_proj + added K/V) | ~5M (low-rank A,B per attn Linear) |
| **Mechanism** | Adds new cross-attn branch reading from CLIP image embeddings | Modifies existing Q/K/V/out projections in place |
| **Conditioning input** | Reference image at inference | None extra — same prompt as base |
| **Merges into base?** | No (always two branches) | Yes, optionally (zero inference overhead) |
| **Per-image extra input at inference** | Yes (reference image) | No |
| **Best at** | "Make this output look like that image" | "Make this output match a learned style/distribution" |
| **What you need on disk** | `image_proj.bin` + cross-attn deltas | `pytorch_lora_weights.safetensors` |
| **In your pipeline** | `adapters/ip_adapter/{model,train}.py` | `adapters/lora/{layers,model,train}.py` |

---

## Glossary

- **`ε_θ(z_t, t, c)`** — the noise-prediction network. *Is* the SDXL UNet. Takes a noisy latent + timestep + conditioning, returns its guess at the noise that was added. Trained by minimizing `MSE(ε, ε_θ(...))`.
- **CLIP-L** — OpenAI ViT-Large CLIP text encoder. Outputs `(B, 77, 768)`.
- **CLIP-G** ("bigG") — OpenCLIP ViT-Giant CLIP text encoder. Outputs `(B, 77, 1280)` + pooled `(B, 1280)`.
- **SDXL `c_text`** — channel-concat of CLIP-L hidden states + CLIP-G hidden states → `(B, 77, 2048)`. That's why SDXL cross-attn `to_k` / `to_v` have `in_features=2048`.
- **`Q`, `K`, `V`** — Query, Key, Value in attention: `softmax(Q·Kᵀ/√d)·V`. `to_q`/`to_k`/`to_v` are the projections that produce them; `to_out` is the post-attention output projection.
- **`add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`** — SDXL's *added* cross-attention projections (used in attn2 alongside the regular ones). Required LoRA targets — dropping them silently breaks SDXL.
- **VAE** — Variational Autoencoder. SDXL trains/runs in latent space (128×128×4); VAE encoder maps pixel→latent at training time, decoder maps latent→pixel at inference.
- **`v`-prediction** — alternative training parameterization to ε-prediction. SDXL base uses ε-prediction; we follow that.

---

## ⚠️ Status notes (as of 2026-04-22)

### LoRA — being rebuilt
Per `docs/lora-implementation-plan.md`. Current state:
- ✅ `adapters/lora/layers.py::LoRALinear` — implemented + tested (`tests/test_lora_layer.py`).
- 🚧 `adapters/lora/model.py` — scaffolded with TODOs for `inject_lora_into_unet`, `save_lora_weights`, `load_lora_weights`. Test skeletons in `tests/test_lora_inject.py` (currently 17 failing on `NotImplementedError` — that's the work board).
- ⏳ `adapters/lora/train.py` — not yet started.

### IP-Adapter — needs critical fixes
The code in `adapters/ip_adapter/{model,train}.py` has the right ingredients but the **IP-Adapter mechanism itself is not wired up**. Three concrete defects:

1. **`_add_ip_attention_layers` is a no-op.** It iterates `unet.attn_processors` and copies them into a dict, then never calls `unet.set_attn_processor(...)`. So no `IPAdapterAttnProcessor` ever gets installed → no `add_k_proj` / `add_v_proj` exist on the UNet → the right side of the IP-Adapter diagram above doesn't exist in the model.

2. **`encode_image` is never called during training.** The training step's UNet call is:
   ```python
   noise_pred = adapter.unet(
       noisy_latents, timesteps,
       encoder_hidden_states=text_embeds,            # text only
       added_cond_kwargs={"text_embeds": pooled_text_embeds,
                          "time_ids": add_time_ids},
   ).sample
   ```
   No `image_prompt_embeds` anywhere. The reference image never enters the loss.

3. **Therefore `image_proj_model` receives no useful gradient.** Its outputs aren't part of the loss-bearing computation graph. Loss will look like it's training (because PyTorch happily runs `optimizer.step()`) but the IP-Adapter weights drift on noise, not signal.

**What a correct IP-Adapter trainer needs:**
- In `model.py`: actually inject `IPAdapterAttnProcessor2_0` (from diffusers) into every cross-attn block via `unet.set_attn_processor(...)`.
- In `train.py`: load a CLIP-preprocessed reference image per sample, run `image_embeds = adapter.image_encoder(ref).image_embeds`, project with `image_proj_model`, concat with text on the token axis, pass the concatenation as `encoder_hidden_states`. The injected `IPAdapterAttnProcessor` splits text vs image internally and runs decoupled cross-attn.
- In `ProductDataset`: add a `clip_pixel_values` field (the reference image preprocessed by `CLIPImageProcessor`).

**Recommended fix path:** mirror diffusers' reference trainer at
`examples/research_projects/ip_adapter/...` rather than re-deriving from scratch.

This is the same class of bug (architecturally aware, structurally broken) that motivated the LoRA rebuild. Worth flagging to the team before any PACE compute is spent.
