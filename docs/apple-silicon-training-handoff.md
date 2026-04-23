# Local Training on macOS Apple Silicon (IP-Adapter / LoRA)

> Author: Jason (IP-Adapter track)
> Date: 2026-04-21
> Audience: teammates running adapter training locally on M-series Macs
> Based on: my end-to-end Shopify IP-Adapter run on M4 Pro 48 GB (3000 steps, ~9 hours)

---

## TL;DR

This doc is a record of how I got IP-Adapter training working, with a focus on **every landmine I hit so you don't have to step on them again**. Key numbers:

- **Hardware**: M1/M2/M3/M4, **unified memory ≥ 48 GB** (32 GB will likely OOM)
- **Wall clock**: ~9 h per platform × 3 platforms ≈ 27 h total (at 512 res, 3000 steps)
- **Precision must be fp32**: MPS mixed precision (fp16/bf16) is fundamentally incompatible with SDXL+adapter. **This is the biggest trap.**
- **Data**: use the team bundle from `share/`; don't re-run curation yourself
- **Smoke first**: run a 5-step smoke before committing to a full 9-hour run

Running SDXL-scale training on Apple Silicon is **workable but finicky**. Follow the steps below and you'll skip every pothole I fell into.

---

## 1. Environment setup

### 1.1 Python 3.11

macOS's bundled `python3` is 3.9.6 — too old (`pyproject` requires ≥ 3.10, modern diffusers/transformers also need 3.10+). Install via Homebrew:

```bash
brew install python@3.11
python3.11 --version    # should print Python 3.11.x
```

### 1.2 Venv + dependencies

```bash
cd /path/to/StudioDiffusion    # your cloned repo root
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt    # ~5–10 minutes
```

**Trap 1: `segment-anything-2` is not on PyPI.**
That line in `requirements.txt` will fail to install. If it breaks the run, install it from GitHub instead:

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Then re-run `pip install -r requirements.txt` for the rest. (It's a dependency of the segmentation track — unrelated to LoRA training, but the broken pin blocks pip either way.)

**Trap 2: `pip install -e .` will fail.**
`pyproject.toml` has a broken `build-backend` value (`setuptools.backends.legacy:build` is not a valid entry point). **Don't try to fix it right now** — there's a `conftest.py` workaround in the repo root that makes pytest and scripts work without the editable install. See §3 trap B below.

### 1.3 accelerate config

Don't run `accelerate config` (the interactive prompts are easy to misanswer). Just write the MPS-specific config file directly:

```bash
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
YAML

accelerate env    # verify the config was picked up
```

> ⚠️ `mixed_precision: 'no'` is load-bearing. **Don't** set this to `fp16` or `bf16`. See §3 trap A for why.

### 1.4 Verify

```bash
python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
# expected: torch 2.x  mps True

pytest tests/test_imports.py -v
# 1 passed
```

If `pytest` errors with `ModuleNotFoundError: No module named 'adapters'`, the repo's `conftest.py` didn't kick in — make sure the (otherwise empty) `conftest.py` at the repo root exists.

---

## 2. Data

**Don't re-run curation.** Use the team bundle:

1. Ask the data track owner for **`StudioDiffusion-datashare-PLATFORM-ONLY-*.tar.gz`** (platform sets only, ~70 MB — the right variant for people who only need to train, not re-curate).
2. Drop it at the repo root.
3. Extract:

```bash
cd /path/to/StudioDiffusion
tar -xzf StudioDiffusion-datashare-PLATFORM-ONLY-*.tar.gz
```

After extraction you should have:

```
data/platform_sets/
├── shopify/      (~800 JPEGs)
├── etsy/         (~754)
├── ebay/         (~800)
└── manifests/
    ├── shopify_train.csv     (353 rows)
    ├── shopify_val.csv       (88)
    ├── etsy_train.csv        (325)
    ├── etsy_val.csv          (81)
    ├── ebay_train.csv        (518)
    └── ebay_val.csv          (129)
```

**Verify the dataset loads** (CPU-only, seconds):

```bash
python -c "
from pathlib import Path
from adapters.ip_adapter.train import ProductDataset
for p in ['shopify', 'etsy', 'ebay']:
    d = Path(f'data/platform_sets/{p}')
    tr = ProductDataset(d, 'train', image_size=512)
    va = ProductDataset(d, 'val', image_size=512)
    print(f'{p}: train={len(tr)} val={len(va)}')
"
```

Expected:
```
shopify: train=353 val=88
etsy: train=325 val=81
ebay: train=518 val=129
```

> **Note:** the `image_path` column in the manifest CSVs is an absolute path on the packager's machine — it won't exist on your disk. `ProductDataset` resolves each row automatically by applying the `curate_platform.py` output naming rule (`<parent>_<basename>` primary, plain basename fallback) under `data/platform_sets/<platform>/`. **You don't need to rewrite paths or patch the CSVs.**

---

## 3. The three big traps on Apple Silicon + SDXL (must read)

I hit all three during smoke and fixed them. You are **very likely** to hit the same ones on the LoRA side.

### Trap A — MPS autocast is incompatible with SDXL

**Symptom:** launching with `accelerate launch ... --mixed_precision fp16` (or `bf16`) crashes on the very first forward pass:

```
failed assertion `Destination NDArray and Accumulator NDArray cannot have
different datatype in MPSNDArrayMatrixMultiplication'
```

or:

```
RuntimeError: Expected query, key, and value to have the same dtype, but got
query.dtype: float key.dtype: c10::Half and value.dtype: c10::Half instead.
```

**Root cause:** PyTorch's MPS backend doesn't correctly handle autocast across the mix of Linear/Conv/SDPA ops in SDXL. Activations get cast to fp16 on the fly while some weights stay fp32, and the matmul destination/accumulator dtypes end up mismatched. Both fp16 and bf16 hit this — it's not a dtype-specific problem, it's autocast on MPS.

**The only stable fix: fp32 end-to-end.** It's not enough to flip the CLI flag — you need all four of these to agree:

1. `scripts/train_*.sh` uses `--mixed_precision no`
2. `configs/base.yaml` has `mixed_precision: "no"` — **must match the CLI**. accelerate's precedence is `Accelerator(mixed_precision=…)` ctor kwarg > CLI flag, so if base.yaml still says fp16, the CLI value is silently ignored.
3. Every `from_pretrained` call in `adapters/*/train.py` explicitly passes `torch_dtype=torch.float32`. Reason: SDXL's `text_encoder_2` and `madebyollin/sdxl-vae-fp16-fix` safetensors on HuggingFace are stored as fp16 by default — without an explicit dtype you get mixed-precision frozen models and the K/V mismatch comes back.
4. No hardcoded `.to(dtype=torch.float16)` in train.py for VAE or `pixel_values`.

**Time cost:** fp32 is ~1.7–2× slower than fp16 on the same hardware. Real measurements on M4 Pro 48 GB:
- 768 res fp32: ~28 s per effective-step (3000 steps ≈ 24 h per platform)
- 512 res fp32: ~11 s per effective-step (3000 steps ≈ 9 h per platform)
- I settled on **512 res** (see trap C).

### Trap B — `accelerate launch` breaks relative imports

**Symptom:**

```
ModuleNotFoundError: No module named 'adapters'
```

**Root cause:** when you run `accelerate launch adapters/lora/train.py`, Python adds only the script's directory (`adapters/lora/`) to `sys.path` — the repo root is not on the path. So the top-of-file `from adapters.lora.model import ...` (or `from adapters.ip_adapter.train import ProductDataset`) can't find the `adapters` package.

**Fix:** prefix `PYTHONPATH="${PWD}:${PYTHONPATH:-}"` in the launcher:

```bash
# scripts/train_lora.sh final form:
PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch \
  --mixed_precision no \
  --num_processes 1 \
  adapters/lora/train.py \
  --config "$CONFIG"
```

### Trap C — 1024 res is too slow / too memory-hungry on M4 Pro

**Symptom** (if you leave `image_size: 1024` in the yamls):
- Without gradient checkpointing → peak memory ~35 GB, the 48 GB machine starts swapping to disk and training becomes effectively frozen
- With gradient checkpointing + fp32 → ~24 h per platform × 3 = 72 h total

**My choice: drop to 512.** Rationale:
- IP-Adapter's CLIP image encoder input is 336² regardless of the diffusion training resolution — the per-platform aesthetic signal is encoded at low res and is **decoupled from the diffusion training resolution**
- 512 brings 3000 steps to ~9 h per platform × 3 = 27 h total
- LoRA likely behaves similarly — start at 512; upgrade later if generation quality is disappointing

**Concrete change:**

```yaml
# configs/lora/{shopify,etsy,ebay}.yaml — the data: block
data:
  platform_dir: "data/platform_sets/<platform>"
  image_size: 512           # ← change this (may default to 1024)
  center_crop: true
```

Also make sure `configs/base.yaml` has `gradient_checkpointing: true` (it should by default) **AND** that `adapters/lora/train.py` actually consumes it — i.e. calls `unet.enable_gradient_checkpointing()` somewhere in `train()`. I had to add that line for IP-Adapter (see commit `33991b4`); the LoRA train.py may need the same wiring.

---

## 4. LoRA-specific preparation

### 4.1 Patch `scripts/train_lora.sh`

The current launcher (pre-IP-Adapter-branch) has both trap A and trap B. Rewrite it to:

```bash
#!/usr/bin/env bash
set -euo pipefail

PLATFORM="${1:-shopify}"
CONFIG="configs/lora/${PLATFORM}.yaml"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "=== Training LoRA for platform: $PLATFORM ==="

# PYTHONPATH: put the repo root on sys.path so `from adapters.lora.model import ...`
#   resolves under `accelerate launch`.
# MPS fallback: a few ops aren't implemented on MPS yet; PyTorch silently routes
#   them to CPU when this env var is set.
# mixed_precision no: MPS autocast is broken for this stack; fp32 is the only
#   stable path on Apple Silicon. See docs/apple-silicon-training-handoff.md §3A.
PYTHONPATH="${PWD}:${PYTHONPATH:-}" PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch \
  --mixed_precision no \
  --num_processes 1 \
  adapters/lora/train.py \
  --config "$CONFIG"

echo "=== Done. Checkpoint saved to checkpoints/lora/$PLATFORM/final ==="
```

### 4.2 Check whether LoRA's `train.py` needs the same fixes

I applied these to `adapters/ip_adapter/train.py`. Each row tells you whether LoRA likely needs the matching change:

| Fix | Purpose | Does LoRA likely need it? |
|---|---|---|
| Add `torch_dtype=torch.float32` to every `from_pretrained` call | Prevents loading fp16 weights by accident | **Very likely yes** |
| Remove hardcoded `vae.to(dtype=torch.float16)` and `pixel_values.to(dtype=torch.float16)` | Keeps the fp32 story consistent end-to-end | **Very likely yes** |
| Call `unet.enable_gradient_checkpointing()` near the top of `train()` | Without it, peak memory is ~30 GB+ | **Check whether LoRA's train.py consumes `cfg.training.gradient_checkpointing`** — if not, add it |
| Build a separate val DataLoader + `_validate()` helper with fixed RNG seed | Makes val loss comparable across steps; without it you're training blind | **Strongly recommended** — see commit `78aebbf` for the full pattern you can copy |

I already updated the LoRA `ProductDataset(...)` call site to match the new signature (commit `6fa50d7`), so `ProductDataset(platform_dir=..., split="train", image_size=...)` in `adapters/lora/train.py` line 77-ish is already correct — you don't need to touch it.

### 4.3 Pre-launch config sanity check

Before the first smoke run, sweep `configs/lora/{shopify,etsy,ebay}.yaml` and `configs/base.yaml`:

```bash
python -c "
from omegaconf import OmegaConf
base = OmegaConf.load('configs/base.yaml')
assert base.training.mixed_precision == 'no', base.training.mixed_precision
assert base.training.gradient_checkpointing == True
assert base.logging.report_to is None  # unless you want wandb
for p in ['shopify','etsy','ebay']:
    cfg = OmegaConf.load(f'configs/lora/{p}.yaml')
    assert cfg.data.image_size <= 768
    print(p, 'OK')
"
```

---

## 5. Smoke test (do this first!)

A full run is 9+ hours. Never launch one blind. Run a 5-step smoke first to validate the pipeline end-to-end.

**Temporarily patch `configs/lora/shopify.yaml`** — the `training:` block:

```yaml
training:
  max_train_steps: 5
  learning_rate: 1.0e-4
  train_batch_size: 2
  validation_steps: 5         # overrides base.yaml's 250
  checkpointing_steps: 5      # overrides base.yaml's 500
```

**Run:**

```bash
rm -rf checkpoints/lora/shopify    # wipe any leftover smoke artifacts
bash scripts/train_lora.sh shopify > /tmp/smoke.log 2>&1
```

**Verify** (expect ~5–10 min; the first run downloads the SDXL weights, ~8 GB total):

```bash
cat checkpoints/lora/shopify/train.log
# should contain two lines like:
#   step=5 val_loss=0.07xxxx wall=xxs n=44
#   step=5 val_loss=0.07xxxx wall=xxs n=44

ls checkpoints/lora/shopify/
# should contain: checkpoint-5/ final/ train.log
```

If the two `step=5` val_loss values match **exactly** to 6 decimal places, your fixed-seed `_validate` is working correctly — that's a free correctness check.

**After smoke passes:** restore the yaml to production values (`max_train_steps: 3000`, delete the `validation_steps` and `checkpointing_steps` overrides), wipe the smoke checkpoints:

```bash
rm -rf checkpoints/lora/shopify
```

---

## 6. Full training + monitoring

### 6.1 Launch

```bash
bash scripts/train_lora.sh shopify 2>&1 | tee /tmp/train_logs/lora_shopify.log
```

(I recommend running in the foreground with `tee` rather than backgrounding. Also: **disable "allow the computer to sleep" in System Settings → Displays → Advanced**, or macOS will suspend the training process.)

### 6.2 Monitor during training

In another terminal:

```bash
# val loss curve (one line appended every 250 steps)
tail -f checkpoints/lora/shopify/train.log

# tqdm progress bar + per-step train loss
tail -f /tmp/train_logs/lora_shopify.log
```

Activity Monitor → Memory tab, watch the python process's Real Memory:
- 512 res fp32 + grad ckpt: normal peak ~15–18 GB
- 1024 res fp32 + grad ckpt: normal peak ~25–30 GB
- If it climbs above 35 GB, worry about swap

### 6.3 What a healthy val loss curve looks like

Here's my actual Shopify IP-Adapter curve from a clean 3000-step run:

```
step=250  val_loss=0.073747
step=500  val_loss=0.073364   ↓
step=750  val_loss=0.073088   ↓
step=1000 val_loss=0.072865   ↓
step=1500 val_loss=0.072577   ↓
step=2000 val_loss=0.072463   ↓ (converged)
step=2500 val_loss=0.072478   ≈
step=3000 val_loss=0.072500   ≈
```

Things to look for:
- **Monotonic decrease** up to around step 2000, then a plateau
- **Absolute improvement of ~1–3%** is in the normal range for adapter training — don't expect a 20% drop; that would actually suggest a bug or overfit
- No sustained rise (that would be an overfitting signal)
- LoRA's curve should show the same shape, with potentially different absolute numbers

---

## 7. Error cheatsheet

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'adapters'` | Launcher is missing `PYTHONPATH=$PWD` | Patch `scripts/train_lora.sh` (§4.1) |
| `ModuleNotFoundError: segment_anything_2` | `pip install -r requirements.txt` failed on that package | `pip install git+https://github.com/facebookresearch/segment-anything-2.git` |
| `failed assertion ... MPSNDArrayMatrixMultiplication` | Mixed precision is still leaking | Check `base.yaml` **and** launcher CLI **and** every `from_pretrained` all say fp32 (§3 trap A) |
| `Expected query, key, value to have the same dtype` | Same as above — autocast still casting something | Same as above |
| `Manifest CSV not found: ...` | Bundle not extracted, or wrong location | Confirm `data/platform_sets/manifests/shopify_train.csv` exists |
| `More than 5% of manifest rows could not be resolved` | Bundle version doesn't match manifest | Sync bundle date with the data owner |
| OOM / swap spikes | Gradient checkpointing probably not actually enabled | Confirm `cfg.training.gradient_checkpointing: true` AND that train.py actually calls `unet.enable_gradient_checkpointing()` |
| Process killed mid-training | macOS memory pressure | Close other memory-hungry apps (Chrome, IDE, Docker) |

---

## 8. References (how I built it)

If you want to dig into my implementation:

- **Design spec**: `docs/superpowers/specs/2026-04-21-ip-adapter-manifest-driven-training-design.md`
  - Read the "Addendum — post-smoke findings" section in particular; the fp32/512 decisions are documented there
- **Implementation plan**: `docs/superpowers/plans/2026-04-21-ip-adapter-manifest-driven-training.md`
- **Key commits** (all on branch `feat/ip-adapter-manifest-driven`):
  - `f6114b3` — Manifest-driven `ProductDataset` rewrite (LoRA already imports this)
  - `78aebbf` — `_validate()` helper and training-loop wiring (LoRA can copy this pattern)
  - `33991b4` — Gradient-checkpointing wire-up
  - `7ad5f7d` — **Complete MPS-compat fix set** (fp32, PYTHONPATH, torch_dtype, etc.) — this is probably the commit diff you'll want most
  - `6fa50d7` — Post-review follow-ups (includes the LoRA call-site signature update)

To see all the IP-Adapter changes:

```bash
git diff main..feat/ip-adapter-manifest-driven -- \
  adapters/ip_adapter/train.py \
  adapters/ip_adapter/model.py \
  scripts/train_ip_adapter.sh \
  configs/base.yaml
```

---

## 9. What I haven't verified (honest caveats)

- **I haven't run LoRA end-to-end on this setup.** The LoRA-specific advice above is extrapolated from "traps IP-Adapter hit, LoRA will almost certainly hit the same" — your smoke run may surface new issues.
- **I haven't run the generation/evaluation suite yet** (`evaluation/`). A decreasing training MSE is necessary but not sufficient; final quality has to come from inference + the CLIP-alignment / DINOv2-fidelity / aesthetic-score metrics.

If your smoke run hits a new error that doesn't match the cheatsheet, ping me — it may be a LoRA-code bug rather than an environment issue, and I'm happy to help debug.

Good luck 🎯
