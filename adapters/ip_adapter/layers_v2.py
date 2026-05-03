"""IP-Adapter v2 layer components."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageProjModelV2(nn.Module):
    """
    Project frozen CLIP image embeddings into SDXL cross-attention tokens.

    Expected mapping:
        image_embeds: (batch, clip_embed_dim)
        return:       (batch, num_tokens, cross_attention_dim)
    """

    def __init__(
        self,
        clip_embed_dim: int = 1024,
        cross_attention_dim: int = 2048,
        num_tokens: int = 16,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if clip_embed_dim <=0: 
            raise ValueError(f"clip_embed_dim must be positive, got {clip_embed_dim}")
        self.clip_embed_dim = clip_embed_dim
        if cross_attention_dim <= 0:
            raise ValueError(f"cross_attention_dim must be positive, got {cross_attention_dim}")
        self.cross_attention_dim = cross_attention_dim
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        self.num_tokens = num_tokens
        if hidden_size is not None and hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = hidden_size

        output_dim = num_tokens * cross_attention_dim

        
        self.proj: nn.Module
        if hidden_size is None:
            self.proj = nn.Linear(clip_embed_dim, output_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(clip_embed_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim),
            )

        
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Return projected image prompt tokens."""
        if image_embeds.ndim !=2:
            raise ValueError(f"Expected image_embeds to have shape (batch, clip_embed_dim), got {image_embeds.shape}")
        
        batch_size, embed_dim = image_embeds.shape
        projected = self.proj(image_embeds)
        projected = projected.view(batch_size, self.num_tokens, self.cross_attention_dim)
        normalized = self.norm(projected)
        return normalized



class IPAttnProcessor2_0V2(nn.Module):
    """
    Cross-attention processor with an added IP-Adapter image K/V branch.

    The normal text cross-attention path should remain unchanged. When
    ip_hidden_states is provided, the query from the latent hidden states should
    attend over image tokens through to_k_ip/to_v_ip, then add the image result
    back into the text-attention result with self.scale.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 16,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size is not None and hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = hidden_size
        if cross_attention_dim <= 0:
            raise ValueError(f"cross_attention_dim must be positive, got {cross_attention_dim}")
        self.cross_attention_dim = cross_attention_dim
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        self.num_tokens = num_tokens

        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        ip_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run text cross-attention plus optional image-token attention."""
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states)
        if hidden_states.ndim == 4:
            batch_size, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, hidden_states.shape[1], hidden_states.shape[0])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        batch_size, seq_len, _ = hidden_states.shape
        
        
        text_k = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        text_v = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        query = attn.to_q(hidden_states)
        inner_dim = text_k.shape[-1]
        head_dim = inner_dim // attn.heads
        input_ndim= hidden_states.ndim
        query = query.view(batch_size, -1, attn.heads,head_dim).transpose(1, 2)
        text_k = text_k.view(batch_size, -1, attn.heads,head_dim).transpose(1, 2)
        text_v = text_v.view(batch_size, -1, attn.heads,head_dim).transpose(1, 2)
        
        scaled_text_attn = torch.nn.functional.scaled_dot_product_attention(query, text_k, text_v, attn_mask=attention_mask)
        
        
        if ip_hidden_states is not None:
            image_k = self.to_k_ip(ip_hidden_states)
            image_v = self.to_v_ip(ip_hidden_states)
            image_k = image_k.view(batch_size, -1, attn.heads,head_dim).transpose(1, 2)
            image_v = image_v.view(batch_size, -1, attn.heads,head_dim).transpose(1, 2)
            scaled_image_attn = torch.nn.functional.scaled_dot_product_attention(query, image_k, image_v, attn_mask=None)
            scaled_text_attn = scaled_text_attn + self.scale * scaled_image_attn
        hidden_states = scaled_text_attn.transpose(1,2).reshape(batch_size, -1 , attn.heads * head_dim)
        
        out = attn.to_out[0](hidden_states)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(1, 2).view(batch_size, channels, height, width)
        if attn.residual_connection:
            out = out + residual
        out = out/attn.rescale_output_factor
        return out
       
