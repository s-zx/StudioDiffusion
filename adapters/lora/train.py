"""
LoRA training loop for SDXL aesthetic adapters.

Per docs/lora-implementation-plan.md File 3.

Usage
-----
python adapters/lora/train.py --config configs/lora/etsy.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from adapters.lora.model import (
    DEFAULT_TARGET_MODULES,
    inject_lora_into_unet,
    save_lora_weights,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PlatformDataset(Dataset):
    """Platform image dataset with optional manifest-driven train/val splits.

    If ``data/platform_sets/manifests/<platform>_<split>.csv`` exists, we use
    the same resolved-on-disk naming rule as the IP-Adapter dataset so train
    and val splits stay consistent across adapter types. If not, we fall back
    to globbing the whole directory (useful for small local smoke runs).
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        platform_dir: str | Path,
        image_size: int,
        prompt_template: str,
        split: str = "train",
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.platform_dir = Path(platform_dir)
        self.prompt = prompt_template
        self.split = split
        self.image_paths = self._load_image_paths()
        if not self.image_paths:
            if split == "val":
                return
            raise FileNotFoundError(
                f"No images found for split={split!r} under {self.platform_dir}"
            )

        # Standard SDXL pre-processing: resize → center-crop → [-1, 1].
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])

    def _load_image_paths(self) -> list[Path]:
        manifest_csv = (
            self.platform_dir.parent / "manifests" / f"{self.platform_dir.name}_{self.split}.csv"
        )
        if not manifest_csv.exists():
            if self.split == "val":
                return []
            return sorted(
                p for p in self.platform_dir.rglob("*") if p.suffix.lower() in self.IMAGE_EXTS
            )

        with open(manifest_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Manifest is empty (header only): {manifest_csv}")

        resolved: list[Path] = []
        unresolved: list[str] = []
        for row in rows:
            src = Path(row["image_path"])
            primary = self.platform_dir / f"{src.parent.name}_{src.name}"
            secondary = self.platform_dir / src.name
            if primary.exists():
                resolved.append(primary)
            elif secondary.exists():
                resolved.append(secondary)
            else:
                unresolved.append(row["image_path"])

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
                f"[PlatformDataset] {self.platform_dir.name}/{self.split}: "
                f"skipped {len(unresolved)} unresolved rows ({miss_rate:.1%})"
            )
        return resolved

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return {"pixel_values": img_tensor, "prompt": self.prompt}


@torch.no_grad()
def _validate(
    unet,
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
    """Run one deterministic validation pass and append val loss to train.log."""
    unet.eval()
    try:
        gen = torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)
        total_loss = 0.0
        n_batches = 0
        wall_start = time.time()
        weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32

        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample(generator=gen)
            latents = latents * vae.config.scaling_factor

            noise = torch.randn(
                latents.shape,
                generator=gen,
                device=accelerator.device,
                dtype=latents.dtype,
            )
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                generator=gen,
                device=accelerator.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            prompts = batch["prompt"]
            tokens_1 = tokenizer_1(
                prompts,
                padding="max_length",
                max_length=tokenizer_1.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            tokens_2 = tokenizer_2(
                prompts,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            encoder1_out = text_encoder_1(tokens_1, output_hidden_states=True)
            encoder2_out = text_encoder_2(tokens_2, output_hidden_states=True)
            text_embeds = torch.cat(
                [encoder1_out.hidden_states[-2], encoder2_out.hidden_states[-2]], dim=-1
            )
            pooled_text_embeds = encoder2_out[0]
            H = W = cfg.data.image_size
            add_time_ids = torch.tensor(
                [[H, W, 0, 0, H, W]] * batch_size,
                dtype=weight_dtype,
                device=accelerator.device,
            )

            noise_pred = accelerator.unwrap_model(unet)(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeds,
                added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": add_time_ids},
            ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        wall = time.time() - wall_start
        line = f"step={global_step} val_loss={mean_loss:.6f} wall={wall:.1f}s n={n_batches}"
        print(f"[validate] {line}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return mean_loss
    finally:
        unet.train()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(cfg_path: str) -> None:
    # ── 1. Config merge ────────────────────────────────────────────────────
    base_config  = OmegaConf.load("configs/base.yaml")
    platform_cfg = OmegaConf.load(cfg_path)
    cfg          = OmegaConf.merge(base_config, platform_cfg)

    # ── 2. Accelerator ─────────────────────────────────────────────────────
    output_dir = Path(cfg.paths.output_dir) / "lora" / cfg.platform
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log = output_dir / "train.log"
    train_log.touch(exist_ok=True)
    proj_cfg = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.logging.report_to,
        project_config=proj_cfg,
    )
    if accelerator.is_main_process:
        print(f"[accel] device={accelerator.device}  fp={accelerator.mixed_precision}")

    # ── 3. Models (CPU first; placement happens in §8) ─────────────────────
    base = cfg.model.base
    tokenizer_1    = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tokenizer_2    = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", torch_dtype=torch.float16)
    vae            = AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=torch.float16)
    unet           = UNet2DConditionModel.from_pretrained(base, subfolder="unet")  
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # Freeze the auxiliaries (UNet gets selectively re-frozen in §4 by inject_lora_into_unet).
    for m in (text_encoder_1, text_encoder_2, vae):
        m.requires_grad_(False)

    # ── 4. Inject LoRA ─────────────────────────────────────────────────────
    
    trainable_params: list[torch.nn.Parameter] = inject_lora_into_unet(
        unet,
        target_modules=cfg.lora.target_modules,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
    )
    # ── 5. Optimizer ───────────────────────────────────────────────────────
    # Resolve LR: prefer cfg.training.learning_rate (platform yamls put it here),
    # else fall back to cfg.optimizer.learning_rate.
    lr = cfg.training.get("learning_rate", None) or cfg.optimizer.learning_rate
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(lr),
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
        eps=float(cfg.optimizer.epsilon),
    )

    # ── 6. LR scheduler ────────────────────────────────────────────────────
    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler.type,                       
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
        num_training_steps=cfg.training.max_train_steps,
    )

    # ── 7. Dataset + DataLoader ────────────────────────────────────────────
    dataset = PlatformDataset(
        platform_dir=cfg.data.platform_dir,
        image_size=cfg.data.image_size,
        prompt_template=cfg.data.prompt_template,
        split="train",
    )
    val_dataset = PlatformDataset(
        platform_dir=cfg.data.platform_dir,
        image_size=cfg.data.image_size,
        prompt_template=cfg.data.prompt_template,
        split="val",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=False,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
    ) if len(val_dataset) > 0 else None

    # ── 8. Prepare with accelerator ────────────────────────────────────────
    prep_items = [unet, optimizer, dataloader, lr_scheduler]
    if val_loader is not None:
        prep_items.append(val_loader)
    prepared = accelerator.prepare(*prep_items)
    unet, optimizer, dataloader, lr_scheduler = prepared[:4]
    if val_loader is not None:
        val_loader = prepared[4]
    # Auxiliaries don't need prepare (frozen + no grads); just push to device + dtype.
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    vae            = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_1 = text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2 = text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # ── 9. Trackers (the call the old code missed) ─────────────────────────
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── 10. Training loop ──────────────────────────────────────────────────
    global_step = 0
    done = False
    steps_per_epoch = math.ceil(len(dataloader) / cfg.training.gradient_accumulation_steps)
    num_epochs = math.ceil(cfg.training.max_train_steps / steps_per_epoch)
    progress = tqdm(total=cfg.training.max_train_steps, disable=not accelerator.is_main_process)

    for epoch in range(num_epochs):
        if done:
            break
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timestamps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timestamps)
                prompts = batch["prompt"]
                tokens_1 = tokenizer_1(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer_1.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(accelerator.device)
                tokens_2 = tokenizer_2(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(accelerator.device)

                with torch.no_grad():
                    encoder1_out = text_encoder_1(tokens_1, output_hidden_states=True)
                    encoder2_out = text_encoder_2(tokens_2, output_hidden_states=True)
                    text_embeds = torch.cat(
                        [encoder1_out.hidden_states[-2], encoder2_out.hidden_states[-2]],
                        dim=-1,
                    )
                    pooled_text_embeds = encoder2_out[0]
                H = W = cfg.data.image_size
                add_time_ids = torch.tensor(
                    [[H, W, 0, 0, H, W]] * latents.shape[0],
                    dtype=weight_dtype, device=accelerator.device,
                )
                if global_step == 0 and accelerator.is_main_process:
                    print(f"text_embeds={text_embeds.shape}  pooled={pooled_text_embeds.shape}  time_ids={add_time_ids.shape}")

                noise_pred = unet(
                    noisy_latents,
                    timestamps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": add_time_ids},
                ).sample

                # ── 11. Loss (with optional min-SNR weighting) ──────────────
                # min-SNR (Hang et al. 2023): re-weight per-sample MSE by
                #   w_t = min(SNR_t, gamma) / SNR_t
                # → down-weights very-low-noise / very-high-noise extremes,
                #   stabilises training, faster convergence.
                if cfg.noise.snr_gamma and cfg.noise.snr_gamma > 0:
                    snr = compute_snr(noise_scheduler, timestamps)            # (B,)
                    gamma = torch.full_like(snr, float(cfg.noise.snr_gamma))
                    mse_weights = torch.minimum(snr, gamma) / snr             # (B,)
                    per_sample = F.mse_loss(
                        noise_pred.float(), noise.float(), reduction="none"
                    ).mean(dim=[1, 2, 3])                                     # (B,)
                    loss = (per_sample * mse_weights).mean()
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix({"loss": float(loss), "step": global_step})
                accelerator.log(
                    {"train/loss": float(loss), "train/lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step,
                )
                if accelerator.is_main_process:
                    with open(train_log, "a", encoding="utf-8") as f:
                        f.write(
                            f"step={global_step} train_loss={float(loss):.6f} "
                            f"lr={lr_scheduler.get_last_lr()[0]:.8f}\n"
                        )
                if (
                    val_loader is not None
                    and global_step % cfg.training.validation_steps == 0
                ):
                    _validate(
                        unet,
                        val_loader,
                        vae,
                        text_encoder_1,
                        text_encoder_2,
                        tokenizer_1,
                        tokenizer_2,
                        noise_scheduler,
                        accelerator,
                        cfg,
                        global_step,
                        train_log,
                    )
                if global_step % cfg.training.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_lora_weights(
                        accelerator.unwrap_model(unet),
                        output_dir / f"checkpoint-{global_step}",
                        rank=cfg.lora.rank,
                        alpha=cfg.lora.alpha,
                        target_modules=list(cfg.lora.target_modules),
                    )
                if global_step >= cfg.training.max_train_steps:
                    done = True
                    break

    if val_loader is not None:
        _validate(
            unet,
            val_loader,
            vae,
            text_encoder_1,
            text_encoder_2,
            tokenizer_1,
            tokenizer_2,
            noise_scheduler,
            accelerator,
            cfg,
            global_step,
            train_log,
        )

    # ── 12. Final save ─────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(
            accelerator.unwrap_model(unet),
            output_dir / "final",
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            target_modules=list(cfg.lora.target_modules),
        )
    accelerator.end_training()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
