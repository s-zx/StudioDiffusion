"""IP-Adapter v2 training entrypoint."""

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
from diffusers.training_utils import compute_snr
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from adapters.ip_adapter.model_v2 import IPAdapterSDXLV2


class ProductDatasetV2(Dataset):
    """Manifest-driven dataset for product images and CLIP pixels."""

    def __init__(
        self,
        platform_dir: str | Path,
        split: str,
        image_size: int,
        center_crop: bool = True,
    ) -> None:
        
        self.platform_dir = Path(platform_dir)
        self.split = split
        self.image_size = image_size
        self.center_crop = center_crop
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        
        self.items: list[Path] = []
        self.items = self._load_items()
        

        self.captions: dict[str, str] = {}
        self.captions = self._load_captions()

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])


        self.clip_transform = transforms.Compose([
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(336) if center_crop else transforms.RandomCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])


    def _load_items(self) -> list[Path]:
        """Resolve manifest rows to local platform image paths."""
        items = []
        unresolved: list[str] = []
        platform = self.platform_dir.name
        manifest_path = self.platform_dir.parent / "manifests" / f"{platform}_{self.split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
        
        with manifest_path.open(newline= "",encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        total_rows = len(manifest)
        if total_rows == 0:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        for row in manifest:
            source_path = Path(row["image_path"])

            candidate_1 = self.platform_dir / f"{source_path.parent.name}_{source_path.name}"
            candidate_2 = self.platform_dir / source_path.name

            if candidate_1.exists():
                items.append(candidate_1)
            elif candidate_2.exists():
                items.append(candidate_2)
            else:
                unresolved.append(row["image_path"])
        if len(unresolved) / total_rows > 0.05:
            sample = "\n  ".join(unresolved[:5])
            raise ValueError(
                f"More than 5% of manifest rows are missing: {len(unresolved)}/{total_rows}\n"
                f"Sample unresolved paths:\n  {sample}"
            )
        return items

    def _load_captions(self) -> dict[str, str]:
        """Load optional per-image captions, falling back later to a generic caption."""
        captions_dir = Path("data/processed/captions") / self.platform_dir.name
        if not captions_dir.exists():
            return {}
        captions = {}
        for caption_file in captions_dir.glob("*.txt"):
            image_id = caption_file.stem
            caption = caption_file.read_text(encoding="utf-8").strip()
            captions[image_id] = caption
        return captions
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        image_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        clip_pixel_values = self.clip_transform(image)
        caption = self.captions.get(image_path.stem, "a product photo")
        return {"pixel_values": pixel_values, "clip_pixel_values": clip_pixel_values, "caption": caption}


def _load_config(config_path: str | Path) -> DictConfig:
    """Merge configs/base.yaml with one v2 platform config."""
    cfg = OmegaConf.load("configs/base.yaml")
    platform_cfg = OmegaConf.load(config_path)
    return OmegaConf.merge(cfg, platform_cfg)


def _dtype_for_device(mixed_precision: str | None, device: torch.device) -> torch.dtype:
    """Choose model dtype from accelerator mixed precision and device type."""
    if device.type in {"cpu", "mps"}:
        return torch.float32
    elif device.type == "cuda":
        if mixed_precision == "bf16":
            return torch.bfloat16
        elif mixed_precision == "fp16":
            return torch.float16
        else:
            raise ValueError(f"Invalid mixed_precision for CUDA: {mixed_precision}")
    else:
        raise ValueError(f"Unsupported device type: {device.type}")


def _learning_rate(cfg: DictConfig) -> float:
    """Prefer training.learning_rate, falling back to optimizer.learning_rate."""
    return float(cfg.training.get("learning_rate", None) or cfg.optimizer.learning_rate)


def _build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    """Build the LR scheduler from cfg.lr_scheduler."""
    from diffusers.optimization import get_scheduler
    num_training_steps = max(1, int(cfg.training.max_train_steps))
    num_warmup_steps = min(
        int(cfg.lr_scheduler.get("num_warmup_steps", 0)),
        max(0, num_training_steps - 1),
    )
    return get_scheduler(
        name=cfg.lr_scheduler.get("type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def _make_time_ids(
    image_size: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build SDXL added time ids: original size, crop coords, target size."""
    return torch.tensor([image_size, image_size, 0, 0, image_size, image_size], dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1)


def _encode_text(
    captions: list[str],
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize captions and return SDXL sequence + pooled text embeddings."""
    inputs_1 = tokenizer_1(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_1.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    inputs_2 = tokenizer_2(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_2.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        encoder_1_output = text_encoder_1(inputs_1, output_hidden_states=True)
        encoder_2_output = text_encoder_2(inputs_2, output_hidden_states=True)
        text_embeds_1 = encoder_1_output.hidden_states[-2]
        text_embeds_2 = encoder_2_output.hidden_states[-2]
        pooled_text_embeds = encoder_2_output[0]
    text_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)
    return text_embeds, pooled_text_embeds


def _loss_with_optional_snr_weighting(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    cfg: DictConfig,
) -> torch.Tensor:
    """Compute diffusion MSE, optionally with min-SNR weighting."""
    if cfg.noise.snr_gamma and cfg.noise.snr_gamma > 0:
        snr = compute_snr(noise_scheduler, timesteps)
        gamma = torch.full_like(snr, float(cfg.noise.snr_gamma))
        weights = torch.minimum(snr, gamma) / snr
        per_sample_loss = F.mse_loss(
            noise_pred.float(), noise.float(), reduction="none"
        ).mean(dim=[1, 2, 3])
        return (per_sample_loss * weights).mean()
    return F.mse_loss(noise_pred.float(), noise.float())


@torch.no_grad()
def _validate(
    adapter: IPAdapterSDXLV2,
    val_loader: DataLoader,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    accelerator: Accelerator,
    cfg: DictConfig,
    global_step: int,
    log_path: Path,
    model_dtype: torch.dtype,
) -> float:
    """Run deterministic validation and append val_loss to train.log."""
    adapter.eval()
    try:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)
        total_loss = 0.0
        num_batches = 0
        wall_start = time.time()

        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=model_dtype)
            clip_pixel_values = batch["clip_pixel_values"].to(accelerator.device)
            batch_size = pixel_values.shape[0]

            latents = vae.encode(pixel_values).latent_dist.sample(generator=generator)
            latents = latents * vae.config.scaling_factor
            noise = torch.randn(
                latents.shape,
                generator=generator,
                device=accelerator.device,
                dtype=latents.dtype,
            )
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                generator=generator,
                device=accelerator.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            text_embeds, pooled_text_embeds = _encode_text(
                batch["caption"],
                tokenizer_1,
                tokenizer_2,
                text_encoder_1,
                text_encoder_2,
                accelerator.device,
            )
            time_ids = _make_time_ids(cfg.data.image_size, batch_size, model_dtype, accelerator.device)
            noise_pred = adapter(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                encoder_hidden_states=text_embeds,
                added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": time_ids},
                clip_pixel_values=clip_pixel_values,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        wall = time.time() - wall_start
        line = f"step={global_step} val_loss={avg_loss:.6f} wall={wall:.1f}s n={num_batches}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        return avg_loss
    finally:
        adapter.train()


def train(config_path: str) -> None:
    """Run IP-Adapter v2 training."""
    cfg = _load_config(config_path)
    output_dir = Path(cfg.paths.output_dir) / "ip_adapter_v2" / cfg.platform
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_path.touch(exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.logging.report_to,
        project_config=project_config,
    )
    model_dtype = _dtype_for_device(cfg.training.mixed_precision, accelerator.device)

    tokenizer_1 = CLIPTokenizer.from_pretrained(cfg.model.base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.model.base, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        cfg.model.base,
        subfolder="text_encoder",
        torch_dtype=model_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        cfg.model.base,
        subfolder="text_encoder_2",
        torch_dtype=model_dtype,
    )
    vae = AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=model_dtype)
    unet = UNet2DConditionModel.from_pretrained(
        cfg.model.base,
        subfolder="unet",
        torch_dtype=model_dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.base, subfolder="scheduler")
    adapter = IPAdapterSDXLV2(
        unet=unet,
        image_encoder_id=cfg.ip_adapter.image_encoder,
        num_tokens=cfg.ip_adapter.num_tokens,
        adapter_scale=cfg.ip_adapter.adapter_scale,
        proj_hidden_size=cfg.ip_adapter.get("proj_hidden_size", None),
        image_encoder_dtype=model_dtype,
    )
    adapter = adapter.to(dtype=model_dtype)
    if cfg.training.gradient_checkpointing:
        adapter.unet.enable_gradient_checkpointing()

    for model in (vae, text_encoder_1, text_encoder_2):
        model.requires_grad_(False)
        model.eval()

    trainable_params = adapter.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=_learning_rate(cfg),
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
    )
    scheduler = _build_scheduler(cfg, optimizer)

    train_dataset = ProductDatasetV2(
        cfg.data.platform_dir,
        split="train",
        image_size=cfg.data.image_size,
        center_crop=cfg.data.center_crop,
    )
    val_dataset = ProductDatasetV2(
        cfg.data.platform_dir,
        split="val",
        image_size=cfg.data.image_size,
        center_crop=cfg.data.center_crop,
    )
    pin_memory = accelerator.device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    num_epochs = math.ceil(cfg.training.max_train_steps / max(steps_per_epoch, 1))
    adapter, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(adapter, optimizer, train_loader, scheduler, val_loader)
    vae = vae.to(accelerator.device, dtype=model_dtype)
    text_encoder_1 = text_encoder_1.to(accelerator.device, dtype=model_dtype)
    text_encoder_2 = text_encoder_2.to(accelerator.device, dtype=model_dtype)

    global_step = 0
    progress = tqdm(
        total=cfg.training.max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    for _epoch in range(num_epochs):
        adapter.train()
        for batch in train_loader:
            with accelerator.accumulate(adapter):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=model_dtype)
                clip_pixel_values = batch["clip_pixel_values"].to(accelerator.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if cfg.training.gradient_checkpointing:
                    noisy_latents.requires_grad_(True)
                text_embeds, pooled_text_embeds = _encode_text(
                    batch["caption"],
                    tokenizer_1,
                    tokenizer_2,
                    text_encoder_1,
                    text_encoder_2,
                    accelerator.device,
                )
                time_ids = _make_time_ids(
                    cfg.data.image_size,
                    batch_size,
                    model_dtype,
                    accelerator.device,
                )
                noise_pred = adapter(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": time_ids},
                    clip_pixel_values=clip_pixel_values,
                )
                loss = _loss_with_optional_snr_weighting(
                    noise_pred,
                    noise,
                    timesteps,
                    noise_scheduler,
                    cfg,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                current_lr = scheduler.get_last_lr()[0]
                progress.set_postfix({"loss": float(loss), "lr": current_lr})
                accelerator.log({"train/loss": float(loss), "train/lr": current_lr}, step=global_step)
                if accelerator.is_main_process:
                    with log_path.open("a", encoding="utf-8") as file:
                        file.write(
                            f"step={global_step} train_loss={float(loss):.6f} "
                            f"lr={current_lr:.8f}\n"
                        )
                if cfg.training.validation_steps and global_step % cfg.training.validation_steps == 0:
                    _validate(
                        adapter=adapter,
                        val_loader=val_loader,
                        vae=vae,
                        text_encoder_1=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        tokenizer_1=tokenizer_1,
                        tokenizer_2=tokenizer_2,
                        noise_scheduler=noise_scheduler,
                        accelerator=accelerator,
                        cfg=cfg,
                        global_step=global_step,
                        log_path=log_path,
                        model_dtype=model_dtype,
                    )
                if accelerator.is_main_process and cfg.training.checkpointing_steps and global_step % cfg.training.checkpointing_steps == 0:
                    accelerator.unwrap_model(adapter).save_pretrained(output_dir / f"step_{global_step}")
                if global_step >= cfg.training.max_train_steps:
                    break
        if global_step >= cfg.training.max_train_steps:
            break

    progress.close()
    _validate(
        adapter=adapter,
        val_loader=val_loader,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        accelerator=accelerator,
        cfg=cfg,
        global_step=global_step,
        log_path=log_path,
        model_dtype=model_dtype,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(adapter).save_pretrained(output_dir / "final")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to v2 platform YAML")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
