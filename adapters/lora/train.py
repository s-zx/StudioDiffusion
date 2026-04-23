"""
LoRA training loop for SDXL aesthetic adapters.

Per docs/lora-implementation-plan.md File 3.

Usage
-----
python adapters/lora/train.py --config configs/lora/etsy.yaml
"""

from __future__ import annotations

import argparse
import math
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
    """Glob `platform_dir` for images; every sample returns the same fixed prompt.

    Aesthetic adapter: we want the model to learn the *visual distribution* of
    a platform, not per-image text→image alignment. So no captions.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, platform_dir: str | Path, image_size: int, prompt_template: str) -> None:
        self.platform_dir = Path(platform_dir)
        self.prompt = prompt_template
        self.image_paths = sorted(
            p for p in self.platform_dir.rglob("*") if p.suffix.lower() in self.IMAGE_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.platform_dir}")

        # Standard SDXL pre-processing: resize → center-crop → [-1, 1].
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        # TODO (you):
        #   1. Open self.image_paths[idx] with PIL, convert to "RGB".
        #   2. Apply self.transform → tensor of shape (3, image_size, image_size).
        #   3. Return {"pixel_values": that_tensor, "prompt": self.prompt}.
        img= Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return {"pixel_values": img_tensor, "prompt": self.prompt}


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
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
    )

    # ── 8. Prepare with accelerator ────────────────────────────────────────
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
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
                # ── Training step ────────────────────────────────────────────
                # TODO (you): the actual diffusion math.
                #
                #   
                #
                #   # 1. VAE encode pixels → latents, scale by vae.config.scaling_factor.
                #   # 2. Sample noise ε ~ N(0, I) with same shape as latents.
                #   # 3. Sample timesteps t ~ U{0, num_train_timesteps-1}.
                #   # 4. noisy_latents = noise_scheduler.add_noise(latents, ε, t)
                #   # 5. Text-encode batch["prompt"] with both tokenizers/text_encoders;
                #   #      concat hidden_states[-2] of each → encoder_hidden_states (B, 77, 2048)
                #   #      pooled_text_embeds = enc2_out[0]
                #   #      add_time_ids = torch.tensor([[H, W, 0, 0, H, W]] * B, device=…)
                #   # 6. noise_pred = unet(noisy_latents, t,
                #   #                      encoder_hidden_states=...,
                #   #                      added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample
                #   # 7. loss = F.mse_loss(noise_pred.float(), ε.float())
                #   #
                #   # ── 11. Optional min-SNR weighting ───────────────────────
                #   #   if cfg.noise.snr_gamma > 0:
                #   #       from diffusers.training_utils import compute_snr
                #   #       snr = compute_snr(noise_scheduler, t)
                #   #       w = torch.minimum(snr, cfg.noise.snr_gamma * torch.ones_like(snr)) / snr
                #   #       per_sample = F.mse_loss(noise_pred.float(), ε.float(), reduction="none").mean([1,2,3])
                #   #       loss = (per_sample * w).mean()
                #   #
                #   # accelerator.backward(loss)
                #   # if accelerator.sync_gradients:
                #   #     accelerator.clip_grad_norm_(trainable_params, 1.0)
                #   # optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents=vae.encode(pixel_values).latent_dist.sample()
                    latents=latents * vae.config.scaling_factor
                noise=torch.randn_like(latents)
                batch_size=latents.shape[0]
                timestamps=torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()

                noisy_latents= noise_scheduler.add_noise(latents, noise, timestamps)
                prompts=batch["prompt"]
                tokens_1 = tokenizer_1(
                    prompts,padding="max_length",
                    max_length=tokenizer_1.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(accelerator.device)
                tokens_2 = tokenizer_2(
                    prompts,padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(accelerator.device)

                with torch.no_grad():
                    encoder1_out=text_encoder_1(tokens_1, output_hidden_states=True)
                    encoder2_out=text_encoder_2(tokens_2, output_hidden_states=True)

                    text_embeds=torch.cat([encoder1_out.hidden_states[-2], encoder2_out.hidden_states[-2]], dim=-1)
                    pooled_text_embeds = encoder2_out[0]
                H = W = cfg.data.image_size
                add_time_ids = torch.tensor(
                    [[H, W, 0, 0, H, W]] * latents.shape[0],
                    dtype=weight_dtype, device=accelerator.device,
                )
                if global_step == 0 and accelerator.is_main_process:
                    print(f"text_embeds={text_embeds.shape}  pooled={pooled_text_embeds.shape}  time_ids={add_time_ids.shape}")

                noise_pred=unet(noisy_latents, timestamps,
                                 encoder_hidden_states=text_embeds,
                                 added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}).sample

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
