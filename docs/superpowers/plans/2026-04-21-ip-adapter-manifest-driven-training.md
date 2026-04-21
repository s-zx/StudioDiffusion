# IP-Adapter manifest-driven training — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make IP-Adapter training (`adapters/ip_adapter/train.py`) consume the curated manifest CSVs as the source of truth for per-platform train/val splits, report validation MSE every `validation_steps`, and fit reliably on M4 Pro 48 GB at 768×768 by enabling gradient checkpointing and unifying precision.

**Architecture:** Single Python module `adapters/ip_adapter/train.py` is the only behavioral code change — its `ProductDataset` is replaced and a `_validate` helper is added. Configs (`configs/base.yaml`, `configs/ip_adapter/*.yaml`) and the launcher (`scripts/train_ip_adapter.sh`) are edited in lockstep. Tests live under `tests/` and exercise only the new Dataset (no GPU needed); the rest is verified by an inline 5-step training smoke run.

**Tech Stack:** Python 3.11, PyTorch (MPS backend), diffusers ≥0.27, transformers ≥0.40, accelerate ≥0.29, omegaconf, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-04-21-ip-adapter-manifest-driven-training-design.md`

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `adapters/ip_adapter/train.py` | Modify | Replace `ProductDataset`; add `_validate()`; build train+val loaders; enable grad ckpt; init file logger |
| `configs/base.yaml` | Modify | `dataloader_num_workers: 0`; `report_to: null` |
| `configs/ip_adapter/shopify.yaml` | Modify | `image_size: 768`; remove `train_split` |
| `configs/ip_adapter/etsy.yaml` | Modify | same |
| `configs/ip_adapter/ebay.yaml` | Modify | same |
| `scripts/train_ip_adapter.sh` | Modify | `--mixed_precision fp16` |
| `tests/test_product_dataset.py` | **Create** | Unit tests for the new manifest-driven Dataset |
| `~/.cache/huggingface/accelerate/default_config.yaml` | Create (env, not repo) | Non-interactive accelerate config for MPS + fp16 |

No new packages. No package layout changes.

---

## Phase A — Environment setup

### Task A1: Install Python 3.11

**Files:**
- None in repo (modifies system / Homebrew state)

- [ ] **Step 1: Verify Python 3.11 is missing**

Run: `which python3.11`
Expected: empty output, exit 1

- [ ] **Step 2: Install via Homebrew**

Run: `brew install python@3.11`
Expected: completes within 2–5 min; final lines mention `python@3.11` keg-only or symlinked.

- [ ] **Step 3: Verify**

Run: `python3.11 --version`
Expected: `Python 3.11.x`

- [ ] **Step 4: Nothing to commit** (system state, not repo).

---

### Task A2: Create venv and install requirements

**Files:**
- Create: `.venv/` (gitignored — not committed)

- [ ] **Step 1: Create venv with Python 3.11**

Run: `python3.11 -m venv .venv`
Expected: `.venv/` directory created; no output.

- [ ] **Step 2: Activate and upgrade pip**

Run: `source .venv/bin/activate && pip install --upgrade pip`
Expected: pip ≥24 reported.

- [ ] **Step 3: Install requirements**

Run: `pip install -r requirements.txt`
Expected: completes in 5–10 min; installs torch, diffusers, transformers, accelerate, pandas, pytest, etc. No error lines.

- [ ] **Step 4: Verify the critical imports**

Run:
```bash
python -c "import torch; import diffusers; import transformers; import accelerate; import pandas; import pytest; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
```
Expected: prints versions and `mps True`.

- [ ] **Step 5: Verify existing tests still pass**

Run: `pytest tests/test_imports.py -v`
Expected: 1 passed.

- [ ] **Step 6: Nothing to commit** (`.venv/` is in `.gitignore`).

---

### Task A3: Write accelerate config

**Files:**
- Create: `~/.cache/huggingface/accelerate/default_config.yaml` (outside repo)

- [ ] **Step 1: Write the config**

Run:
```bash
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
Expected: file written, no output.

- [ ] **Step 2: Verify accelerate sees it**

Run: `accelerate env`
Expected: prints config including `mixed_precision: fp16`, `distributed_type: NO`, `use_cpu: False`. No errors.

- [ ] **Step 3: Nothing to commit** (config is in user home, not repo).

---

## Phase B — `ProductDataset` (TDD)

> **Pattern reminder:** Each task in this phase follows TDD strictly: write the failing test first, run it to confirm it fails for the *right* reason, then write only enough code to make it pass.

### Task B1: Happy path — train and val splits load correctly

**Files:**
- Create: `tests/test_product_dataset.py`
- Modify: `adapters/ip_adapter/train.py:45-90` (replace `ProductDataset` class body)

- [ ] **Step 1: Write the failing test**

Create `tests/test_product_dataset.py`:

```python
"""Unit tests for the manifest-driven ProductDataset.

These tests assume the team data bundle has been extracted at the repo root
(see share/TEAM_DATA_BUNDLE_README.txt). They run on CPU and finish in
seconds — no model weights or GPU needed.
"""
from pathlib import Path

import pytest
import torch

from adapters.ip_adapter.train import ProductDataset


PLATFORM_DIR = Path("data/platform_sets/shopify")
IMAGE_SIZE = 768


@pytest.fixture(scope="module")
def shopify_train():
    return ProductDataset(PLATFORM_DIR, split="train", image_size=IMAGE_SIZE)


@pytest.fixture(scope="module")
def shopify_val():
    return ProductDataset(PLATFORM_DIR, split="val", image_size=IMAGE_SIZE)


def test_train_split_loads(shopify_train):
    assert len(shopify_train) > 200, f"expected >200 train items, got {len(shopify_train)}"


def test_val_split_loads(shopify_val):
    assert len(shopify_val) > 50, f"expected >50 val items, got {len(shopify_val)}"


def test_item_shape_and_keys(shopify_train):
    item = shopify_train[0]
    assert set(item.keys()) == {"pixel_values", "clip_pixel_values", "caption"}
    assert isinstance(item["pixel_values"], torch.Tensor)
    assert item["pixel_values"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert item["clip_pixel_values"].shape == (3, 336, 336)
    assert isinstance(item["caption"], str) and len(item["caption"]) > 0
```

- [ ] **Step 2: Run, confirm it fails**

Run: `pytest tests/test_product_dataset.py -v`
Expected: FAIL — current `ProductDataset.__init__(self, data_dir, image_size=1024)` doesn't accept `split=`. The first error will be `TypeError: __init__() got an unexpected keyword argument 'split'` or similar (depending on pytest invocation order).

- [ ] **Step 3: Replace `ProductDataset` in `adapters/ip_adapter/train.py`**

Open `adapters/ip_adapter/train.py`. At the top of the file, add `import csv` to existing imports (after `import math`):

```python
import csv
```

Replace the entire `ProductDataset` class (lines ~45–90 in the current file) with:

```python
class ProductDataset(Dataset):
    """Manifest-driven product image dataset for IP-Adapter training.

    Reads `data/platform_sets/manifests/<platform>_<split>.csv` and resolves
    each row's `image_path` to a local file under `<platform_dir>/`, using
    the same naming rule that `data/curate_platform.py` writes
    (`<parent>_<basename>` on collision, plain `<basename>` otherwise).

    Captions are looked up at `data/processed/captions/<platform>/<stem>.txt`
    if present; otherwise fall back to `"a product photo"`.
    """

    def __init__(self, platform_dir: Path, split: str, image_size: int = 768) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.platform_dir = Path(platform_dir)
        self.split = split
        self.image_size = image_size

        manifest_csv = (
            self.platform_dir.parent / "manifests" / f"{self.platform_dir.name}_{split}.csv"
        )
        if not manifest_csv.exists():
            raise FileNotFoundError(
                f"Manifest CSV not found: {manifest_csv}\n"
                f"Expected layout: data/platform_sets/manifests/<platform>_<split>.csv"
            )

        with open(manifest_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Manifest is empty (header only): {manifest_csv}")

        self.items: list[dict] = []
        unresolved: list[str] = []
        for row in rows:
            src = Path(row["image_path"])
            primary = self.platform_dir / f"{src.parent.name}_{src.name}"
            secondary = self.platform_dir / src.name
            if primary.exists():
                resolved = primary
            elif secondary.exists():
                resolved = secondary
            else:
                unresolved.append(row["image_path"])
                continue
            self.items.append({
                "path": resolved,
                "category": (row.get("category") or "").strip(),
            })

        miss_rate = len(unresolved) / len(rows)
        if miss_rate > 0.05:
            sample = "\n  ".join(unresolved[:5])
            raise RuntimeError(
                f"More than 5% of manifest rows could not be resolved to disk "
                f"({len(unresolved)}/{len(rows)} = {miss_rate:.1%}). "
                f"Bundle and manifest may be from different curate runs.\n"
                f"Sample unresolved paths:\n  {sample}"
            )
        if unresolved:
            print(
                f"[ProductDataset] {self.platform_dir.name}/{split}: "
                f"skipped {len(unresolved)} unresolved rows ({miss_rate:.1%})"
            )

        # Captions (optional)
        caption_dir = Path("data/processed/captions") / self.platform_dir.name
        self.captions: dict[str, str] = {}
        if caption_dir.exists():
            for txt in caption_dir.glob("*.txt"):
                self.captions[txt.stem] = txt.read_text().strip()

        # Transforms — diffusion branch (image_size×image_size) and CLIP branch (336×336)
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_transform = transforms.Compose([
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        item = self.items[idx]
        image = Image.open(item["path"]).convert("RGB")
        caption = self.captions.get(item["path"].stem, "a product photo")
        return {
            "pixel_values": self.transform(image),
            "clip_pixel_values": self.clip_transform(image),
            "caption": caption,
        }
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_product_dataset.py -v`
Expected: 3 passed (`test_train_split_loads`, `test_val_split_loads`, `test_item_shape_and_keys`).

- [ ] **Step 5: Commit**

```bash
git add adapters/ip_adapter/train.py tests/test_product_dataset.py
git commit -m "feat(ip-adapter): manifest-driven ProductDataset

Replaces directory-rglob loading with CSV manifest parsing
(data/platform_sets/manifests/<platform>_<split>.csv). Resolves
each row's image_path to disk via the curate_platform.py output
naming rule (<parent>_<basename> with bare basename fallback).

Adds basic happy-path tests under tests/test_product_dataset.py
covering train/val sizes and item dict shape."
```

---

### Task B2: Path resolution — both naming forms work

**Files:**
- Modify: `tests/test_product_dataset.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_product_dataset.py`:

```python
def test_resolves_primary_form(shopify_train):
    """At least one item should resolve via the <parent>_<basename> form
    (e.g. '08_08dcf907.jpg', 'shard_00004_000476.jpg')."""
    assert any("_" in it["path"].name for it in shopify_train.items), \
        "expected at least one primary-form (<parent>_<basename>) resolution"


def test_resolves_secondary_form(shopify_train):
    """At least one item should resolve via the plain basename form
    (e.g. '000023.jpg', 'cd39cd67.jpg')."""
    assert any("_" not in it["path"].name for it in shopify_train.items), \
        "expected at least one plain-basename resolution"


def test_all_resolved_paths_exist(shopify_train, shopify_val):
    for ds in (shopify_train, shopify_val):
        for item in ds.items:
            assert item["path"].exists(), f"missing on disk: {item['path']}"
```

- [ ] **Step 2: Run, expect pass**

Run: `pytest tests/test_product_dataset.py::test_resolves_primary_form tests/test_product_dataset.py::test_resolves_secondary_form tests/test_product_dataset.py::test_all_resolved_paths_exist -v`
Expected: 3 passed. The current `data/platform_sets/shopify/` bundle confirmedly contains both forms (verified during brainstorming), so these assertions hold without any fallback handling.

- [ ] **Step 3: Commit**

```bash
git add tests/test_product_dataset.py
git commit -m "test(ip-adapter): assert both filename forms resolve in ProductDataset"
```

---

### Task B3: Error-handling tests

**Files:**
- Modify: `tests/test_product_dataset.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_product_dataset.py`:

```python
def test_invalid_split_raises():
    with pytest.raises(ValueError, match="split must be"):
        ProductDataset(PLATFORM_DIR, split="test", image_size=IMAGE_SIZE)


def test_missing_manifest_raises(tmp_path):
    fake_platform = tmp_path / "fake_platform"
    fake_platform.mkdir()
    (tmp_path / "manifests").mkdir()
    # train CSV deliberately not created
    with pytest.raises(FileNotFoundError, match="Manifest CSV not found"):
        ProductDataset(fake_platform, split="train", image_size=IMAGE_SIZE)


def test_empty_manifest_raises(tmp_path):
    platform_dir = tmp_path / "fake_platform"
    platform_dir.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "fake_platform_train.csv").write_text("image_path,category\n")
    with pytest.raises(ValueError, match="Manifest is empty"):
        ProductDataset(platform_dir, split="train", image_size=IMAGE_SIZE)


def test_excessive_misses_raises(tmp_path):
    platform_dir = tmp_path / "fake_platform"
    platform_dir.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    rows = ["image_path,category"]
    for i in range(20):
        rows.append(f"/nonexistent/x/y_{i}.jpg,FAKE")
    (manifests / "fake_platform_train.csv").write_text("\n".join(rows) + "\n")
    with pytest.raises(RuntimeError, match="More than 5%"):
        ProductDataset(platform_dir, split="train", image_size=IMAGE_SIZE)
```

- [ ] **Step 2: Run, expect pass**

Run: `pytest tests/test_product_dataset.py -v`
Expected: all 10 tests pass (3 from B1 + 3 from B2 + 4 from B3).

If any fail because the implementation needs adjustment, fix the implementation in `adapters/ip_adapter/train.py` until all tests pass, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_product_dataset.py
git commit -m "test(ip-adapter): error handling for ProductDataset (split, missing/empty manifest, miss-rate)"
```

---

## Phase C — Training loop changes

### Task C1: Add `_validate` helper and wire train+val loaders into `train()`

**Files:**
- Modify: `adapters/ip_adapter/train.py` (multiple regions)

- [ ] **Step 1: Add `time` import and the `_validate` helper above `train()`**

In `adapters/ip_adapter/train.py`, add `import time` near the other stdlib imports (after `import math`).

Insert this function just above the existing `def train(cfg_path: str)`:

```python
@torch.no_grad()
def _validate(
    adapter,
    val_loader,
    vae,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    noise_scheduler,
    accelerator,
    cfg,
    global_step: int,
    log_path: Path,
) -> float:
    """Run one full pass over the val loader and log mean MSE.

    Uses a fresh torch.Generator seeded with cfg.training.seed every call,
    so noise + timestep sequences are identical across validation runs and
    val/loss curves are directly comparable.
    """
    adapter.eval()
    try:
        gen = torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)
        total_loss = 0.0
        n_batches = 0
        wall_start = time.time()

        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
            clip_pixel_values = batch["clip_pixel_values"].to(accelerator.device, dtype=torch.float32)

            latents = vae.encode(pixel_values).latent_dist.sample(generator=gen)
            latents = latents * vae.config.scaling_factor

            noise = torch.randn(
                latents.shape, generator=gen,
                device=accelerator.device, dtype=latents.dtype,
            )
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), generator=gen, device=accelerator.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            tokens_1 = tokenizer_1(
                batch["caption"], padding="max_length",
                max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)
            tokens_2 = tokenizer_2(
                batch["caption"], padding="max_length",
                max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)

            enc1_out = text_encoder_1(tokens_1, output_hidden_states=True)
            enc2_out = text_encoder_2(tokens_2, output_hidden_states=True)
            text_embeds = torch.cat(
                [enc1_out.hidden_states[-2], enc2_out.hidden_states[-2]], dim=-1,
            )
            pooled_text_embeds = enc2_out[0]
            add_time_ids = torch.tensor(
                [[cfg.data.image_size, cfg.data.image_size, 0, 0,
                  cfg.data.image_size, cfg.data.image_size]] * bsz,
                dtype=torch.float32, device=accelerator.device,
            )

            image_prompt_embeds, _ = accelerator.unwrap_model(adapter).encode_image(
                clip_pixel_values
            )
            noise_pred = accelerator.unwrap_model(adapter).unet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embeds,
                added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": add_time_ids},
                cross_attention_kwargs={"ip_hidden_states": image_prompt_embeds},
            ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        wall = time.time() - wall_start
        line = f"step={global_step} val_loss={mean_loss:.6f} wall={wall:.1f}s n={n_batches}"
        print(f"[validate] {line}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        return mean_loss
    finally:
        adapter.train()
```

- [ ] **Step 2: Replace dataset/loader construction in `train()`**

Find the existing block in `train()`:

```python
dataset = ProductDataset(
    data_dir=Path(cfg.data.platform_dir),
    image_size=cfg.data.image_size,
)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.training.train_batch_size,
    shuffle=True,
    num_workers=cfg.training.dataloader_num_workers,
    pin_memory=False,  # MPS does not support pinned memory
)
```

Replace with:

```python
platform_dir = Path(cfg.data.platform_dir)
train_dataset = ProductDataset(platform_dir, split="train", image_size=cfg.data.image_size)
val_dataset = ProductDataset(platform_dir, split="val", image_size=cfg.data.image_size)
print(f"[data] {platform_dir.name}: train={len(train_dataset)} val={len(val_dataset)}")

dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.training.train_batch_size,
    shuffle=True,
    num_workers=cfg.training.dataloader_num_workers,
    pin_memory=False,  # MPS does not support pinned memory
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.training.train_batch_size,
    shuffle=False,
    num_workers=0,            # val pass is short — no workers needed
    pin_memory=False,
)
```

- [ ] **Step 3: Prepare val loader through accelerator and define `train_log`**

Find the existing `accelerator.prepare(...)` line:

```python
adapter, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    adapter, optimizer, dataloader, lr_scheduler
)
```

Replace with:

```python
adapter, optimizer, dataloader, lr_scheduler, val_loader = accelerator.prepare(
    adapter, optimizer, dataloader, lr_scheduler, val_loader
)
train_log = output_dir / "train.log"
```

- [ ] **Step 4: Wire `_validate` into the training loop and final**

Inside the training loop, find:

```python
if accelerator.sync_gradients:
    progress_bar.update(1)
    global_step += 1
    progress_bar.set_postfix({"loss": loss.item(), "step": global_step})

    if global_step % cfg.training.checkpointing_steps == 0:
        if accelerator.is_main_process:
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            accelerator.unwrap_model(adapter).save_pretrained(ckpt_dir)

    if global_step >= cfg.training.max_train_steps:
        break
```

Replace with (adds the `_validate` call):

```python
if accelerator.sync_gradients:
    progress_bar.update(1)
    global_step += 1
    progress_bar.set_postfix({"loss": loss.item(), "step": global_step})

    if global_step % cfg.training.checkpointing_steps == 0:
        if accelerator.is_main_process:
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            accelerator.unwrap_model(adapter).save_pretrained(ckpt_dir)

    if global_step % cfg.training.validation_steps == 0:
        _validate(
            adapter, val_loader, vae,
            text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2,
            noise_scheduler, accelerator, cfg, global_step, train_log,
        )

    if global_step >= cfg.training.max_train_steps:
        break
```

Then after the outer `for epoch` loop and `accelerator.wait_for_everyone()`, before `accelerator.unwrap_model(adapter).save_pretrained(output_dir / "final")`, add a final validation:

```python
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    _validate(
        adapter, val_loader, vae,
        text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2,
        noise_scheduler, accelerator, cfg, global_step, train_log,
    )
    accelerator.unwrap_model(adapter).save_pretrained(output_dir / "final")
accelerator.end_training()
```

- [ ] **Step 5: Quick sanity import**

Run: `pytest tests/test_imports.py -v && pytest tests/test_product_dataset.py -v`
Expected: 1 + 10 passed. (Confirms train.py still imports cleanly after edits.)

- [ ] **Step 6: Commit**

```bash
git add adapters/ip_adapter/train.py
git commit -m "feat(ip-adapter): add validation pass and wire train+val loaders

Adds _validate() helper that runs the full val set with fixed-seed
noise+timestep RNG so val/loss is comparable across training steps.
Train loop calls it every cfg.training.validation_steps and once
after training. Loss line is appended to checkpoints/ip_adapter/<platform>/train.log."
```

---

### Task C2: Bundled fixes — gradient checkpointing, drop tracker, file logger init

**Files:**
- Modify: `adapters/ip_adapter/train.py` (3 small regions)

- [ ] **Step 1: Enable gradient checkpointing on the wrapped UNet**

In `train()`, find the line:

```python
adapter = IPAdapterSDXL(
    unet=unet,
    image_encoder_id=cfg.ip_adapter.image_encoder,
    num_tokens=cfg.ip_adapter.num_tokens,
    adapter_scale=cfg.ip_adapter.adapter_scale,
)
```

**Immediately after** that block, add:

```python
if cfg.training.gradient_checkpointing:
    # Enable on the unet AFTER IPAdapterSDXL has injected its cross-attn
    # processors so they're also covered by checkpointing.
    adapter.unet.enable_gradient_checkpointing()
```

- [ ] **Step 2: Initialize file log directory at the top of `train()`**

Just after `output_dir = Path(cfg.paths.output_dir) / "ip_adapter" / cfg.platform`, add:

```python
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "train.log").touch(exist_ok=True)
```

(The `_validate` helper also calls `parent.mkdir(...)` defensively, but creating it up-front means the file exists from step 0 even before the first val.)

- [ ] **Step 3: Verify logging tracker is now disabled at config level**

Confirm `cfg.logging.report_to` is `None` after Phase D edits to `base.yaml`. The existing `Accelerator(... log_with=cfg.logging.report_to ...)` will receive `log_with=None`, which means no tracker — no further code change needed in train.py for this fix.

(No code edit in this step — this step is just a sanity reminder. Phase D is what actually flips the YAML.)

- [ ] **Step 4: Sanity check imports**

Run: `pytest tests/test_imports.py tests/test_product_dataset.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add adapters/ip_adapter/train.py
git commit -m "fix(ip-adapter): enable gradient checkpointing + ensure log dir exists

Calls adapter.unet.enable_gradient_checkpointing() when configured —
without it, SDXL forward+backward at 768/1024 res peaks ~30+ GB on
M4 Pro and OOMs/swaps. Also pre-creates output_dir and train.log so
the first val log line has a stable target."
```

---

## Phase D — Config and launcher edits

### Task D1: `configs/base.yaml`

**Files:**
- Modify: `configs/base.yaml`

- [ ] **Step 1: Edit `dataloader_num_workers` and `report_to`**

In `configs/base.yaml`:

Change `dataloader_num_workers: 4` to `dataloader_num_workers: 0`.
Change `report_to: "wandb"` to `report_to: null`.

Resulting `training:` and `logging:` sections should look like:

```yaml
training:
  seed: 42
  mixed_precision: "fp16"          # bf16 on Ampere+
  gradient_checkpointing: true
  dataloader_num_workers: 0
  train_batch_size: 4
  gradient_accumulation_steps: 4   # effective batch = 16
  max_train_steps: 5000
  checkpointing_steps: 500
  validation_steps: 250
```

```yaml
logging:
  project: "studio-diffusion"
  report_to: null
```

- [ ] **Step 2: No commit yet** (bundle with platform yaml + launcher edits in D3).

---

### Task D2: Three platform yamls — `image_size: 768`, drop `train_split`

**Files:**
- Modify: `configs/ip_adapter/shopify.yaml`
- Modify: `configs/ip_adapter/etsy.yaml`
- Modify: `configs/ip_adapter/ebay.yaml`

- [ ] **Step 1: For each of the three files, edit the `data:` block**

Original block (identical across all three):

```yaml
data:
  platform_dir: "data/platform_sets/<platform>"
  train_split: 0.8
  image_size: 1024
  center_crop: true
```

New block:

```yaml
data:
  platform_dir: "data/platform_sets/<platform>"
  image_size: 768
  center_crop: true
```

(`<platform>` differs per file — leave that line unchanged. Just remove `train_split: 0.8` and change `image_size: 1024` → `image_size: 768`.)

Apply to:
- `configs/ip_adapter/shopify.yaml`
- `configs/ip_adapter/etsy.yaml`
- `configs/ip_adapter/ebay.yaml`

- [ ] **Step 2: No commit yet** (bundle with launcher edit in D3).

---

### Task D3: Launcher — `--mixed_precision fp16`

**Files:**
- Modify: `scripts/train_ip_adapter.sh`

- [ ] **Step 1: Edit the accelerate launch command**

Find this in `scripts/train_ip_adapter.sh`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  adapters/ip_adapter/train.py \
  --config "$CONFIG"
```

Change `--mixed_precision bf16` to `--mixed_precision fp16`. Final block:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch \
  --mixed_precision fp16 \
  --num_processes 1 \
  adapters/ip_adapter/train.py \
  --config "$CONFIG"
```

- [ ] **Step 2: Commit all D-phase config edits together**

```bash
git add configs/base.yaml configs/ip_adapter/shopify.yaml configs/ip_adapter/etsy.yaml configs/ip_adapter/ebay.yaml scripts/train_ip_adapter.sh
git commit -m "config(ip-adapter): 768-res, manifest-driven splits, fp16, no tracker

- base.yaml: num_workers 0 (MPS-friendly), report_to null (logging disabled)
- platform yamls: image_size 1024 -> 768; remove dead train_split field
  (manifest is now the source of truth for splits)
- launcher: mixed_precision bf16 -> fp16 (matches base.yaml + train.py
  hardcoded VAE/pixel fp16; bf16 on MPS is less mature)"
```

---

## Phase E — Smoke tests

### Task E1: Dataset smoke (CPU only, < 1 minute)

**Files:**
- None modified.

- [ ] **Step 1: Run the per-platform dataset smoke**

Run:
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
Expected: three lines like `shopify: train=320 val=80 OK`. No errors. (Numbers will be approximate — anywhere ≥ 200/50 is fine.)

If a platform fails resolution >5%, investigate the bundle vs the manifest before proceeding.

---

### Task E2: 5-step training smoke (downloads SDXL on first run, ~30 min)

**Files:**
- Temporarily modify: `configs/ip_adapter/shopify.yaml` (revert at end)

- [ ] **Step 1: Patch shopify.yaml for a 5-step run**

Edit `configs/ip_adapter/shopify.yaml`. In the `training:` block, override:

```yaml
training:
  max_train_steps: 5
  learning_rate: 1.0e-4
  train_batch_size: 2
```

Add (or override) `validation_steps: 5` and `checkpointing_steps: 5` in the same `training:` block — these are inherited from base by default; per-platform override:

```yaml
training:
  max_train_steps: 5
  learning_rate: 1.0e-4
  train_batch_size: 2
  validation_steps: 5
  checkpointing_steps: 5
```

- [ ] **Step 2: Launch the 5-step run**

Run: `bash scripts/train_ip_adapter.sh shopify 2>&1 | tee /tmp/smoke.log`

Expected during execution:
- Hugging Face downloads (first time only): SDXL UNet, text encoders, VAE, CLIP image encoder. ~8 GB total. ~10–30 min on broadband.
- Then: `[data] shopify: train=~320 val=~80`
- Progress bar reaches step 5
- One `[validate] step=5 val_loss=... wall=...s n=~40` line at the end
- Process exits cleanly (no traceback)

- [ ] **Step 3: Verify outputs**

```bash
ls checkpoints/ip_adapter/shopify/
cat checkpoints/ip_adapter/shopify/train.log
ls checkpoints/ip_adapter/shopify/checkpoint-5/
ls checkpoints/ip_adapter/shopify/final/
```
Expected:
- `train.log` exists, contains exactly one line starting with `step=5 val_loss=`
- `checkpoint-5/` directory exists with `image_proj_model.pt` and `ip_attn_processors.pt`
- `final/` directory exists with the same two files

- [ ] **Step 4: Check peak memory**

Open Activity Monitor → Memory tab during training. Real Memory of the python process should peak below 30 GB at 768 res. If significantly above, recheck that gradient_checkpointing is actually enabled (check stdout for any related warning, or add a `print(adapter.unet.is_gradient_checkpointing)` after the enable call).

- [ ] **Step 5: Revert smoke yaml changes**

Edit `configs/ip_adapter/shopify.yaml`, restore `training:` block to the production values:

```yaml
training:
  max_train_steps: 3000
  learning_rate: 1.0e-4
  train_batch_size: 2
```

(Remove the `validation_steps` and `checkpointing_steps` override lines — they fall back to base.yaml's 250 / 500.)

- [ ] **Step 6: Clean up smoke checkpoints**

Run: `rm -rf checkpoints/ip_adapter/shopify/`

- [ ] **Step 7: Commit nothing** (smoke runs are not artifacts; yaml revert returns file to its post-D2 state — no diff to commit).

If you accidentally committed the smoke override before the revert, do `git diff HEAD` to confirm the diff is back to clean, otherwise commit the revert.

---

## After this plan

The remaining work — full Shopify / Etsy / eBay training runs — is operational, not code work. Each is one command from a clean baseline:

```bash
bash scripts/train_ip_adapter.sh shopify
bash scripts/train_ip_adapter.sh etsy
bash scripts/train_ip_adapter.sh ebay
```

Track progress via:
- `tail -f checkpoints/ip_adapter/<platform>/train.log` (val loss curve)
- Activity Monitor (peak memory)
- `ls -la checkpoints/ip_adapter/<platform>/` (intermediate checkpoint cadence)

Estimated wall-clock at 768-res, 3000 steps: ~8 hours per platform on M4 Pro 48 GB.

---

## Self-review notes

- **Spec coverage**: all six design sections (Dataset, Validation, Bug fixes, Config edits, Env setup, Verification) map to phases A–E with explicit tasks. ✓
- **No placeholders**: every code block is concrete; every command has expected output; every commit message is written out. ✓
- **Type / signature consistency**: `ProductDataset(platform_dir, split, image_size)` signature appears identically in tests, smoke, and `train()` integration. `_validate(...)` signature matches both call sites (in-loop and final). ✓
- **TDD where it makes sense**: Phase B uses strict TDD for `ProductDataset` (the only piece with isolated logic worth unit-testing). The `_validate` helper and bug fixes are integration-tested via the 5-step smoke (Phase E2) — pure-function unit tests would require mocking SDXL, which is too heavy for the scope.
- **One open dependency**: `manifest_csv` is the convention `platform_dir.parent / "manifests" / "<name>_<split>.csv"`. If `curate_platform.py` ever changes its output layout, this convention must be revisited — flagged as a future risk in the spec.
