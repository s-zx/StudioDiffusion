from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from adapters.ip_adapter.model import ImageProjModel


def test_image_proj_model_default_projection_is_backward_compatible() -> None:
    proj = ImageProjModel(clip_embed_dim=6, cross_attention_dim=10, num_tokens=4)
    image_embeds = torch.randn(3, 6)

    tokens = proj(image_embeds)

    assert isinstance(proj.proj, nn.Linear)
    assert tokens.shape == (3, 4, 10)


def test_image_proj_model_accepts_hidden_projection_size() -> None:
    proj = ImageProjModel(
        clip_embed_dim=6,
        cross_attention_dim=10,
        num_tokens=4,
        hidden_size=14,
    ).to(dtype=torch.bfloat16)
    image_embeds = torch.randn(3, 6, dtype=torch.bfloat16)

    tokens = proj(image_embeds)

    assert isinstance(proj.proj, nn.Sequential)
    assert tokens.shape == (3, 4, 10)
    assert tokens.dtype == torch.bfloat16


def test_image_proj_model_rejects_invalid_hidden_projection_size() -> None:
    with pytest.raises(ValueError):
        ImageProjModel(clip_embed_dim=6, cross_attention_dim=10, num_tokens=4, hidden_size=0)