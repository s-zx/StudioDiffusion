# IP-Adapter manifest-driven training — design

**Date**: 2026-04-21
**Owner**: Jason
**Hardware target**: MacBook Pro M4 Pro, 48 GB unified memory

## Context

Jason owns IP-Adapter training for three platforms (Shopify / Etsy / eBay) on the StudioDiffusion project. The teammate-shared data bundle (`StudioDiffusion-datashare-PLATFORM-ONLY-20260421.tar.gz`) has been extracted at the repo root. Each platform directory holds ~800 curated JPEGs, but only ~400 of them per platform are documented in the train/val manifest CSVs at `data/platform_sets/manifests/<platform>_{train,val}.csv` — the rest are leftovers from an earlier curate run with unknown flags.

The current `adapters/ip_adapter/train.py`:

- Loads images by `Path.rglob("*")` over the platform directory, ignoring manifests
- Does not honor the configured 80/20 train/val split
- Has no validation pass, so the configured `validation_steps: 250` is a dead knob
- Does not call `unet.enable_gradient_checkpointing()`, so training peaks ~35 GB at 1024×1024 — would swap or OOM on M4 Pro 48 GB
- Mixes precision modes inconsistently (launcher requests bf16, base.yaml says fp16, train.py hardcodes VAE/pixels to fp16)
- Configures `report_to: wandb` but never calls `accelerator.init_trackers()` — logging silently no-ops

## Goal

Make IP-Adapter training fit and run reliably on M4 Pro 48 GB, using a proper train/val split from manifests, within a realistic wall-clock budget (~25 hours total for all three platforms at 768×768).

## Non-goals

- LoRA track (owned by teammate; separate pipeline)
- BLIP-2 caption generation (training falls back to `"a product photo"` if missing — acceptable since IP-Adapter signal comes through the CLIP image branch)
- Category-aware training / stratified sampling (future ablation; `category` field is preserved for it but not consumed)
- Hyperparameter sweep automation (YAML has a `hyperparameter_sweep` block but no runner; YAGNI)
- Multi-GPU / distributed (single MPS device)
- W&B / TensorBoard logging (Jason opted out — local file logging only)

## Design

### 1. Dataset — manifest-driven

`ProductDataset` in `adapters/ip_adapter/train.py` is replaced.

Constructor:

```python
ProductDataset(platform_dir: Path, split: Literal["train", "val"], image_size: int)
```

Manifest path (hardcoded convention, matches `data/curate_platform.py:444-446`):

```
platform_dir.parent / "manifests" / f"{platform_dir.name}_{split}.csv"
```

Each CSV row has `image_path` (absolute, from packager's machine) and `category` (may be empty).

**Path resolution to local disk**, mirroring the curate output naming rule (`data/curate_platform.py:434-437`):

```python
src = Path(row["image_path"])
primary   = platform_dir / f"{src.parent.name}_{src.name}"  # collision-fallback form
secondary = platform_dir / src.name                          # default form
```

**Resolution policy**:

- `primary` exists → use `primary`
- else `secondary` exists → use `secondary`
- else → skip row, increment missed counter
- If `missed / total > 0.05` after parsing the whole manifest → `RuntimeError` listing the first 5 unresolved `image_path` values
- If manifest is empty (header only) → `ValueError`
- If manifest CSV does not exist → `FileNotFoundError` with the expected path

`__getitem__` returns the same dict keys as today: `pixel_values`, `clip_pixel_values`, `caption`. `category` is held on the instance (`self.items[idx]["category"]`) for future use but not returned in batches.

Caption lookup unchanged: `data/processed/captions/<platform_dir.name>/<resolved_stem>.txt`, silently falling back to `"a product photo"`.

Image transforms unchanged: 768×768 LANCZOS resize + center crop + ToTensor + Normalize([0.5],[0.5]) for the diffusion branch; 336×336 BICUBIC + CLIP normalization for the IP-Adapter branch.

### 2. Validation pass

New private function in `train.py`:

```python
@torch.no_grad()
def _validate(
    adapter, val_loader, vae,
    text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2,
    noise_scheduler, accelerator, cfg, global_step,
) -> float
```

**Behavior**:

- `adapter.eval()` on entry; `adapter.train()` on exit (in `finally`)
- For deterministic comparable val loss across calls: a fresh `torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)` is created per call and used for both the noise tensor and the timestep sample. Same RNG every call ⇒ same noise/timestep sequence ⇒ `val/loss` is comparable across training steps.
- Loss is the same MSE on epsilon prediction as training
- Iterates the full val DataLoader (~80 items / platform; ~40 batches at bs=2; ~2 min per call on M4 Pro)
- Returns mean loss; appends a one-line entry to `checkpoints/ip_adapter/<platform>/train.log` of the form: `step=<n> val_loss=<f> wall=<sec>`
- Also `print()` the same line so it shows in the terminal

**Trigger**:

- In the training loop after `progress_bar.update(...)`: `if global_step > 0 and global_step % cfg.training.validation_steps == 0`
- Once after the training loop exits, before saving the final checkpoint

### 3. Bundled fixes (required to make training fit M4 Pro 48 GB)

| # | Fix | Location | Why |
|---|---|---|---|
| 3.1 | Call `unet.enable_gradient_checkpointing()` | `train()` after `IPAdapterSDXL(...)` construction (so adapter-injected cross-attn modules are also covered) and before `Accelerator.prepare` | Without it, 1024-res SDXL peaks ~35 GB on M4 Pro → swap/OOM. At 768 + ckpt, peak ≈ 15–18 GB. |
| 3.2 | Unify precision to fp16 | `scripts/train_ip_adapter.sh`: `--mixed_precision fp16` (was `bf16`); `base.yaml` already says `fp16` | MPS bf16 is less battle-tested; train.py hardcodes VAE/pixels fp16; keep everything fp16 to avoid silent fallbacks |
| 3.3 | `dataloader_num_workers: 0` | `base.yaml` (was 4) | MPS + macOS `fork` multiprocessing is flaky; SDXL training is GPU-bound — workers gain ~nothing |
| 3.4 | Disable logging tracker | `base.yaml`: `report_to: null`; `train()`: drop `log_with=` from `Accelerator(...)` and never call `init_trackers` | Jason opted out of wandb/tb |

### 4. Config edits

**`configs/base.yaml`**:

- `training.dataloader_num_workers: 0` (was 4)
- `logging.report_to: null` (was `wandb`)

**`configs/ip_adapter/{shopify,etsy,ebay}.yaml`** (all three):

- Remove `data.train_split: 0.8` (manifest split is now authoritative; field is dead)
- `data.image_size: 768` (was 1024)

**`scripts/train_ip_adapter.sh`**:

- `--mixed_precision fp16` (was `bf16`)

### 5. Environment setup

To be run after code changes, before first training run:

```bash
# Python 3.11 venv
python3.11 -m venv .venv
source .venv/bin/activate

# Dependencies (~5–10 min)
pip install -r requirements.txt

# accelerate config — write directly, no interactive prompt
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
YAML
```

First training launch downloads model weights into `~/.cache/huggingface/`: SDXL UNet (~5 GB), text encoders 1+2 (~1.7 GB), VAE fp16-fix (~0.2 GB), CLIP ViT-L/14-336 image encoder (~0.6 GB), plus tokenizer / scheduler configs — **~8 GB total** on disk. Subsequent runs reuse the cache.

No `wandb` / `tensorboard` / HF token setup required.

### 6. Verification (before launching real training)

**Step A — dataset smoke** (no GPU, no model weights):

```bash
python -c "
from pathlib import Path
from adapters.ip_adapter.train import ProductDataset
for p in ['shopify', 'etsy', 'ebay']:
    d = Path(f'data/platform_sets/{p}')
    tr = ProductDataset(d, 'train', image_size=768)
    va = ProductDataset(d, 'val', image_size=768)
    assert len(tr) > 200 and len(va) > 50, (p, len(tr), len(va))
    item = tr[0]
    assert item['pixel_values'].shape == (3, 768, 768)
    assert item['clip_pixel_values'].shape == (3, 336, 336)
    print(f'{p}: train={len(tr)} val={len(va)} OK')
"
```

**Step B — 5-step training smoke**:

Temporarily set `max_train_steps: 5, validation_steps: 5, checkpointing_steps: 5` for `shopify.yaml`, run `bash scripts/train_ip_adapter.sh shopify`, confirm:

- Process exits cleanly
- `checkpoints/ip_adapter/shopify/train.log` contains exactly one `step=5 val_loss=...` line
- `checkpoints/ip_adapter/shopify/checkpoint-5/` and `checkpoints/ip_adapter/shopify/final/` exist
- Peak resident memory (Activity Monitor) stayed below 30 GB

Then revert smoke values and proceed to full Shopify run.

## Risks / open questions

- **MPS fp16 numerical stability for SDXL VAE**: known hazard. Mitigated by `madebyollin/sdxl-vae-fp16-fix` checkpoint already in `base.yaml`. No expected issue.
- **Wall-clock estimate (±30%)**: 2–3 sec / micro-batch at 768-res is based on community M4 Pro benchmarks. Real number revealed by the smoke test. If substantially slower, the cheap fallbacks are `image_size: 512` or reduce `max_train_steps` from 3000 to 2000.
- **Manifest-to-disk mismatch tolerance (5%)**: conservative guardrail. The bundle as packaged shows 100% match (verified during brainstorming), so this is for future-proofing against bundle drift.
- **Captions are constant fallback strings**: `"a product photo"` is text-conditioning-constant across the dataset. IP-Adapter training works regardless. If results are weak after Shopify, generating real captions via `data/generate_captions.py` is a follow-up.
- **400 train / 80 val per platform may feel small**: with effective batch 8 and 3000 steps = 24,000 image iterations = ~75 epochs. For a low-rank adapter learning a platform aesthetic, this is in the normal range; risk is over-fitting to val, mitigated by the val curve being visible.

## Acceptance criteria

1. `ProductDataset(platform_dir, split="train"|"val", image_size=768)` loads ≥ 200 train / ≥ 50 val per platform from manifest CSVs with no `FileNotFoundError`
2. 5-step smoke training completes; `train.log` contains one `step=5 val_loss=…` line
3. Full Shopify training run (3000 steps) completes within 12 hours, producing `checkpoints/ip_adapter/shopify/final/` plus intermediate checkpoints every 500 steps
4. Peak resident memory during training stays below 40 GB

## Out-of-scope follow-ups (not part of this spec)

- BLIP-2 caption generation pass
- Category-stratified sampling using the `category` column
- Inference / generation pipeline integration
- Hyperparameter sweep runner
