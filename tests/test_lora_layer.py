"""
Unit tests for `adapters.lora.layers.LoRALinear`.

These tests are CPU-only, fast (~0.5s total), and require only torch.
They validate the contract from docs/lora-implementation-plan.md (File 1):

  - Forward broadcasts over arbitrary leading batch/sequence dims.
  - At init, output equals base(x) (because lora_B = 0).
  - Only lora_A and lora_B are trainable; the wrapped base is frozen.
  - Gradients flow into LoRA params and NOT into base.weight / base.bias.
  - Bad rank raises ValueError.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adapters.lora import LoRALinear


@pytest.fixture
def layer() -> LoRALinear:
    torch.manual_seed(0)
    return LoRALinear(nn.Linear(64, 128), rank=4, alpha=8.0, dropout=0.0)


def test_forward_shape_broadcasts(layer: LoRALinear) -> None:
    x = torch.randn(2, 10, 64)
    y = layer(x)
    assert y.shape == (2, 10, 128)


def test_forward_shape_2d(layer: LoRALinear) -> None:
    """Also works for plain (batch, features) inputs."""
    x = torch.randn(5, 64)
    y = layer(x)
    assert y.shape == (5, 128)


def test_starts_identical_to_base(layer: LoRALinear) -> None:
    """lora_B = 0 at init → output must equal base(x) exactly."""
    x = torch.randn(2, 10, 64)
    y_lora = layer(x)
    y_base = layer.base(x)
    assert torch.allclose(y_lora, y_base, atol=1e-6)


def test_only_lora_params_trainable(layer: LoRALinear) -> None:
    trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}


def test_lora_branch_active_after_perturbation(layer: LoRALinear) -> None:
    """Once lora_B != 0, output must diverge from base(x)."""
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        layer.lora_B.normal_(std=0.1)
    assert not torch.allclose(layer(x), layer.base(x))


def test_grads_flow_only_into_lora(layer: LoRALinear) -> None:
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        layer.lora_B.normal_(std=0.1)  # need nonzero so backward isn't trivial
    layer.zero_grad()
    layer(x).sum().backward()
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert layer.base.weight.grad is None
    assert layer.base.bias.grad is None


def test_invalid_rank_raises() -> None:
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(8, 8), rank=0)
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(8, 8), rank=-1)


def test_scaling_value() -> None:
    layer = LoRALinear(nn.Linear(8, 8), rank=4, alpha=16.0)
    assert layer.scaling == pytest.approx(4.0)
