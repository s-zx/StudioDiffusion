"""Tests for the IP-Adapter v2 layer implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adapters.ip_adapter.layers_v2 import ImageProjModelV2, IPAttnProcessor2_0V2


class ToyAttention(nn.Module):
    """Tiny stand-in for diffusers Attention used by processor tests."""

    def __init__(self, hidden_size: int = 8, cross_attention_dim: int = 12, heads: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = False
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Identity()])

    def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
        return attention_mask


def test_image_proj_shape_and_dtype() -> None:
    proj = ImageProjModelV2(
        clip_embed_dim=6,
        cross_attention_dim=10,
        num_tokens=4,
        hidden_size=14,
    ).to(dtype=torch.bfloat16)
    image_embeds = torch.randn(3, 6, dtype=torch.bfloat16)

    tokens = proj(image_embeds)

    assert tokens.shape == (3, 4, 10)
    assert tokens.dtype == torch.bfloat16


def test_image_proj_rejects_bad_shape() -> None:
    proj = ImageProjModelV2(clip_embed_dim=6, cross_attention_dim=10, num_tokens=4)
    with pytest.raises(ValueError):
        proj(torch.randn(3, 2, 6))


def test_ip_attention_zero_image_tokens_match_no_image_tokens() -> None:
    torch.manual_seed(0)
    attn = ToyAttention()
    proc = IPAttnProcessor2_0V2(hidden_size=8, cross_attention_dim=12, num_tokens=4)
    hidden_states = torch.randn(2, 5, 8)
    text_states = torch.randn(2, 7, 12)

    without_image = proc(attn, hidden_states, encoder_hidden_states=text_states)
    with_zero_image = proc(
        attn,
        hidden_states,
        encoder_hidden_states=text_states,
        ip_hidden_states=torch.zeros(2, 4, 12),
    )

    assert torch.allclose(without_image, with_zero_image, atol=1e-6)


def test_ip_attention_nonzero_image_tokens_change_output() -> None:
    torch.manual_seed(0)
    attn = ToyAttention()
    proc = IPAttnProcessor2_0V2(hidden_size=8, cross_attention_dim=12, num_tokens=4)
    hidden_states = torch.randn(2, 5, 8)
    text_states = torch.randn(2, 7, 12)
    image_states = torch.randn(2, 4, 12)

    without_image = proc(attn, hidden_states, encoder_hidden_states=text_states)
    with_image = proc(
        attn,
        hidden_states,
        encoder_hidden_states=text_states,
        ip_hidden_states=image_states,
    )

    assert not torch.allclose(without_image, with_image)
