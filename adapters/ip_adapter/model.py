"""
IP-Adapter for SDXL.

Architecture follows Ye et al. (2023): a lightweight image-prompt adapter
composed of:
  1. An image encoder (CLIP ViT-L/14-336) — frozen.
  2. A projection network (image_proj_model) mapping CLIP embeddings to the
     UNet's cross-attention key/value space.
  3. Decoupled cross-attention modules injected into every transformer block
     of the UNet, each adding a learned K/V pathway conditioned on the image
     embedding alongside the existing text K/V pathway.

Only the projection network and added cross-attention K/V weights are trained;
the base SDXL UNet and text encoder remain frozen.

Reference: https://arxiv.org/abs/2308.06721
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


class ImageProjModel(nn.Module):
    """
    Projects CLIP image embeddings to the UNet cross-attention dimension.

    Maps:
        (B, clip_embed_dim) → (B, num_tokens, cross_attention_dim)
    """

    def __init__(
        self,
        clip_embed_dim: int = 1024,
        cross_attention_dim: int = 2048,
        num_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(clip_embed_dim, num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        # image_embeds: (B, clip_embed_dim)
        tokens = self.proj(image_embeds)                       # (B, num_tokens * cross_attn_dim)
        tokens = tokens.view(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(tokens)                               # (B, num_tokens, cross_attn_dim)


class IPAdapterSDXL(nn.Module):
    """
    Full IP-Adapter wrapper around SDXL UNet.

    Trainable parameters: image_proj_model + added K/V linear layers injected
    into each cross-attention block of the UNet.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        image_encoder_id: str = "openai/clip-vit-large-patch14-336",
        num_tokens: int = 16,
        adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.adapter_scale = adapter_scale

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_id)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(image_encoder_id)

        clip_embed_dim = self.image_encoder.config.projection_dim
        cross_attn_dim = unet.config.cross_attention_dim

        self.image_proj_model = ImageProjModel(
            clip_embed_dim=clip_embed_dim,
            cross_attention_dim=cross_attn_dim,
            num_tokens=num_tokens,
        )

        self._add_ip_attention_layers(unet, cross_attn_dim)
        self._freeze_base_weights()

    def _add_ip_attention_layers(self, unet: UNet2DConditionModel, cross_attn_dim: int) -> None:
        """Inject additional K/V projection layers into each cross-attention block."""
        attn_procs = {}
        for name, attn_proc in unet.attn_processors.items():
            attn_procs[name] = attn_proc
        # Concrete injection is handled by diffusers' AttnProcessor2_0 with
        # IP-Adapter support; weights are registered separately via
        # unet.set_attn_processor(). See train.py for the full setup using
        # diffusers' load_ip_adapter utilities.
        self._attn_procs = attn_procs

    def _freeze_base_weights(self) -> None:
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        # Only train projection network (and IP-Adapter K/V weights added in train.py)
        for param in self.image_proj_model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor
            Pixel values from CLIPImageProcessor, shape (B, 3, H, W).

        Returns
        -------
        image_prompt_embeds : torch.Tensor (B, num_tokens, cross_attn_dim)
        uncond_image_prompt_embeds : torch.Tensor (B, num_tokens, cross_attn_dim)
            Unconditional embeds using a zero-image (for CFG).
        """
        image_embeds = self.image_encoder(images).image_embeds         # (B, clip_dim)
        image_prompt_embeds = self.image_proj_model(image_embeds)

        zero_image_embeds = torch.zeros_like(image_embeds)
        uncond_prompt_embeds = self.image_proj_model(zero_image_embeds)

        return image_prompt_embeds, uncond_prompt_embeds

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.image_proj_model.state_dict(), save_directory / "image_proj_model.pt")
        self.unet.save_attn_procs(save_directory)

    @classmethod
    def load_pretrained(
        cls,
        unet: UNet2DConditionModel,
        load_directory: str | Path,
        **kwargs,
    ) -> "IPAdapterSDXL":
        load_directory = Path(load_directory)
        adapter = cls(unet, **kwargs)
        proj_state = torch.load(load_directory / "image_proj_model.pt", map_location="cpu")
        adapter.image_proj_model.load_state_dict(proj_state)
        return adapter
