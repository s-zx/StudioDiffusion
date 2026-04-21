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
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
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


class IPAttnProcessor2_0(nn.Module):
    """
    Cross-attention processor with additional IP-Adapter K/V projections.

    Replaces AttnProcessor2_0 in every cross-attention block of the UNet.
    Adds a second attention path (to_k_ip / to_v_ip) conditioned on the
    image prompt embeddings produced by ImageProjModel.

    ip_hidden_states is injected via cross_attention_kwargs at UNet call time.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 16,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        ip_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Standard text cross-attention (memory-efficient SDPA)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # IP-Adapter image cross-attention — only when embeddings are provided
        if ip_hidden_states is not None:
            ip_key   = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key   = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_out = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_out = ip_out.to(query.dtype)
            hidden_states = hidden_states + self.scale * ip_out

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class IPAdapterSDXL(nn.Module):
    """
    Full IP-Adapter wrapper around SDXL UNet.

    Trainable parameters: image_proj_model + to_k_ip/to_v_ip weights in each
    IPAttnProcessor2_0 injected into the UNet's cross-attention blocks.
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
        self.num_tokens = num_tokens

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_block_hidden_size(self, unet: UNet2DConditionModel, name: str) -> int:
        """Return the attention hidden dim for a given attn processor name."""
        if name.startswith("mid_block"):
            return unet.config.block_out_channels[-1]
        block_id = int(name.split(".")[1])
        if name.startswith("up_blocks"):
            return list(reversed(unet.config.block_out_channels))[block_id]
        # down_blocks or any other prefix
        return unet.config.block_out_channels[block_id]

    def _add_ip_attention_layers(
        self, unet: UNet2DConditionModel, cross_attn_dim: int
    ) -> None:
        """
        Replace every cross-attention processor with IPAttnProcessor2_0.
        Self-attention blocks keep the standard AttnProcessor2_0.
        """
        attn_procs: dict = {}
        for name in unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                # Self-attention — no image conditioning
                attn_procs[name] = AttnProcessor2_0()
            else:
                # Cross-attention — inject IP-Adapter K/V pathway
                hidden_size = self._get_block_hidden_size(unet, name)
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attn_dim,
                    num_tokens=self.num_tokens,
                    scale=self.adapter_scale,
                )
        unet.set_attn_processor(attn_procs)

    def _freeze_base_weights(self) -> None:
        # Freeze everything in the UNet (including IP processors registered as children)
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        # Unfreeze only trainable components
        for param in self.image_proj_model.parameters():
            param.requires_grad = True
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, IPAttnProcessor2_0):
                for param in proc.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(
        self, clip_pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode reference images into IP-Adapter conditioning tokens.

        CLIP encoder runs without gradients (frozen).
        image_proj_model participates in the autograd graph during training.

        Returns
        -------
        image_prompt_embeds  : (B, num_tokens, cross_attn_dim)
        uncond_prompt_embeds : (B, num_tokens, cross_attn_dim)  — zero-image CFG baseline
        """
        with torch.no_grad():
            image_embeds = self.image_encoder(clip_pixel_values).image_embeds  # (B, clip_dim)

        image_prompt_embeds = self.image_proj_model(image_embeds)
        uncond_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
        return image_prompt_embeds, uncond_prompt_embeds

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.image_proj_model.state_dict(), save_directory / "image_proj_model.pt")

        ip_state: dict[str, torch.Tensor] = {}
        for name, proc in self.unet.attn_processors.items():
            if isinstance(proc, IPAttnProcessor2_0):
                for k, v in proc.state_dict().items():
                    ip_state[f"{name}.{k}"] = v
        torch.save(ip_state, save_directory / "ip_attn_processors.pt")

    @classmethod
    def load_pretrained(
        cls,
        unet: UNet2DConditionModel,
        load_directory: str | Path,
        **kwargs,
    ) -> "IPAdapterSDXL":
        load_directory = Path(load_directory)
        adapter = cls(unet, **kwargs)
        adapter.image_proj_model.load_state_dict(
            torch.load(load_directory / "image_proj_model.pt", map_location="cpu")
        )
        ip_state = torch.load(load_directory / "ip_attn_processors.pt", map_location="cpu")
        for name, proc in adapter.unet.attn_processors.items():
            if isinstance(proc, IPAttnProcessor2_0):
                prefix = f"{name}."
                proc.load_state_dict(
                    {k[len(prefix):]: v for k, v in ip_state.items() if k.startswith(prefix)}
                )
        return adapter
