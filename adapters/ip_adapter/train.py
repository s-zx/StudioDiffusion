"""
IP-Adapter training loop.

Loss: standard diffusion noise prediction MSE (epsilon-prediction):
    L = E[||eps - eps_theta(z_t, t, c_text, c_image)||^2]

Only the IP-Adapter projection layers and added cross-attention K/V weights
are updated. The SDXL UNet, VAE, and text encoders remain frozen.

Usage
-----
python adapters/ip_adapter/train.py \
    --config configs/ip_adapter/shopify.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from adapters.ip_adapter.model import IPAdapterSDXL


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProductDataset(Dataset):
    """Simple dataset: loads (image, caption, mask) triples from a directory."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, data_dir: Path, image_size: int = 1024) -> None:
        self.data_dir = data_dir
        self.image_paths = sorted(
            p for p in data_dir.rglob("*") if p.suffix.lower() in self.IMAGE_EXTS
        )
        caption_dir = Path("data/processed/captions") / data_dir.name
        self.captions: dict[str, str] = {}
        if caption_dir.exists():
            for txt in caption_dir.glob("*.txt"):
                self.captions[txt.stem] = txt.read_text().strip()

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        caption = self.captions.get(path.stem, "a product photo")
        return {"pixel_values": self.transform(image), "caption": caption}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg_path: str) -> None:
    base_cfg = OmegaConf.load("configs/base.yaml")
    platform_cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(base_cfg, platform_cfg)

    output_dir = Path(cfg.paths.output_dir) / "ip_adapter" / cfg.platform
    proj_cfg = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.logging.report_to,
        project_config=proj_cfg,
    )

    # ---- Models ----
    tokenizer_1 = CLIPTokenizer.from_pretrained(cfg.model.base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.model.base, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(cfg.model.base, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(cfg.model.base, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(cfg.model.vae)
    unet = UNet2DConditionModel.from_pretrained(cfg.model.base, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.base, subfolder="scheduler")

    adapter = IPAdapterSDXL(
        unet=unet,
        image_encoder_id=cfg.ip_adapter.image_encoder,
        num_tokens=cfg.ip_adapter.num_tokens,
        adapter_scale=cfg.ip_adapter.adapter_scale,
    )

    for m in [vae, text_encoder_1, text_encoder_2]:
        m.requires_grad_(False)

    trainable_params = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
    )

    dataset = ProductDataset(
        data_dir=Path(cfg.data.platform_dir),
        image_size=cfg.data.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
    )

    num_epochs = math.ceil(
        cfg.training.max_train_steps
        / math.ceil(len(dataloader) / cfg.training.gradient_accumulation_steps)
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.max_train_steps - cfg.lr_scheduler.num_warmup_steps,
        eta_min=0,
    )

    adapter, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        adapter, optimizer, dataloader, lr_scheduler
    )
    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder_1 = text_encoder_1.to(accelerator.device)
    text_encoder_2 = text_encoder_2.to(accelerator.device)

    global_step = 0
    progress_bar = tqdm(total=cfg.training.max_train_steps, disable=not accelerator.is_local_main_process)

    for epoch in range(num_epochs):
        adapter.train()
        for batch in dataloader:
            with accelerator.accumulate(adapter):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)

                # Encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning (SDXL uses pooled + sequence embeddings)
                tokens_1 = tokenizer_1(
                    batch["caption"], padding="max_length",
                    max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt"
                ).input_ids.to(accelerator.device)
                tokens_2 = tokenizer_2(
                    batch["caption"], padding="max_length",
                    max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
                ).input_ids.to(accelerator.device)

                enc1_out = text_encoder_1(tokens_1, output_hidden_states=True)
                enc2_out = text_encoder_2(tokens_2, output_hidden_states=True)
                text_embeds = torch.cat(
                    [enc1_out.hidden_states[-2], enc2_out.hidden_states[-2]], dim=-1
                )
                pooled_text_embeds = enc2_out[0]
                add_time_ids = torch.tensor(
                    [[cfg.data.image_size, cfg.data.image_size, 0, 0,
                      cfg.data.image_size, cfg.data.image_size]] * bsz,
                    dtype=torch.float32, device=accelerator.device
                )

                # Predict noise
                noise_pred = accelerator.unwrap_model(adapter).unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": add_time_ids},
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

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

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(adapter).save_pretrained(output_dir / "final")
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to platform config YAML")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
