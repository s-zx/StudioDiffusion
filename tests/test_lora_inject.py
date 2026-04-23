"""
Tests for `adapters.lora.model` — LoRA injection, save, and load.

Strategy: build a tiny `nn.Module` whose dotted submodule names mimic the SDXL
UNet's attention structure (e.g. `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`).
This validates injection / save / load logic without ever loading real SDXL weights.

Each test will fail on `NotImplementedError` until the corresponding function in
`adapters/lora/model.py` is implemented. Run with:

    pytest tests/test_lora_inject.py -v

To work on one helper at a time:
    pytest tests/test_lora_inject.py -v -k "name_matches"
    pytest tests/test_lora_inject.py -v -k "inject"
    pytest tests/test_lora_inject.py -v -k "roundtrip"
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from adapters.lora import LoRALinear
from adapters.lora.model import (
    DEFAULT_TARGET_MODULES,
    inject_lora_into_unet,
    load_lora_weights,
    save_lora_weights,
)
from adapters.lora.model import (
    _get_parent,
    _name_matches_target,
    _set_submodule,
)


# ---------------------------------------------------------------------------
# Toy "UNet" mimicking the SDXL submodule layout
# ---------------------------------------------------------------------------

class ToyAttention(nn.Module):
    """Mimics diffusers' Attention module: separate Q/K/V Linears + to_out=ModuleList."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        # SDXL-style: to_out is a ModuleList [Linear, Dropout]
        self.to_out = nn.ModuleList([nn.Linear(d, d), nn.Dropout(0.0)])
        # SDXL added cross-attn projections (only on attn2; we add them everywhere
        # in the toy for test simplicity — the suffix-match logic still applies)
        self.add_q_proj = nn.Linear(d, d)
        self.add_k_proj = nn.Linear(d, d)
        self.add_v_proj = nn.Linear(d, d)
        self.to_add_out = nn.Linear(d, d)


class ToyTransformerBlock(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.attn1 = ToyAttention(d)  # self-attn
        self.attn2 = ToyAttention(d)  # cross-attn


class ToyAttentionsContainer(nn.Module):
    """Mimics `attentions` — a container with `transformer_blocks: ModuleList`."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([ToyTransformerBlock(d)])


class ToyDownBlock(nn.Module):
    """Mimics one SDXL down block: holds an `attentions: ModuleList`."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([ToyAttentionsContainer(d)])


class ToyUNet(nn.Module):
    """Two down blocks; produces dotted paths matching SDXL's structure."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([ToyDownBlock(d) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cheap forward so backward() works for gradient tests.
        attn = _first_attn(self)
        return attn.to_out[0](attn.to_q(x))


@pytest.fixture
def unet() -> ToyUNet:
    torch.manual_seed(0)
    return ToyUNet(d=32)


def _first_attn(unet: ToyUNet) -> ToyAttention:
    """Type-safe accessor to the first attn1 (avoids Pyright false-positives on ModuleList)."""
    down: ToyDownBlock = unet.down_blocks[0]  # type: ignore[assignment]
    cont: ToyAttentionsContainer = down.attentions[0]  # type: ignore[assignment]
    block: ToyTransformerBlock = cont.transformer_blocks[0]  # type: ignore[assignment]
    return block.attn1


# ---------------------------------------------------------------------------
# Helper tests — small and surgical
# ---------------------------------------------------------------------------

class TestNameMatches:
    def test_exact_leaf_match(self) -> None:
        assert _name_matches_target("attn1.to_q", {"to_q"})

    def test_dotted_target_match(self) -> None:
        # "to_out.0" must match the full suffix, not just the leaf "0"
        assert _name_matches_target("attn1.to_out.0", {"to_out.0"})

    def test_does_not_match_dropout_at_to_out_1(self) -> None:
        # The Dropout at to_out.1 should NOT match "to_out.0"
        assert not _name_matches_target("attn1.to_out.1", {"to_out.0"})

    def test_does_not_match_partial(self) -> None:
        # "to_q_extra" must not match "to_q" — suffix matching, not substring
        assert not _name_matches_target("attn1.to_q_extra", {"to_q"})

    def test_no_targets(self) -> None:
        assert not _name_matches_target("attn1.to_q", set())


class TestGetParent:
    def test_simple_attribute(self, unet: ToyUNet) -> None:
        path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        parent, leaf = _get_parent(unet, path)
        assert leaf == "to_q"
        assert isinstance(parent, ToyAttention)
        assert getattr(parent, leaf) is _first_attn(unet).to_q

    def test_modulelist_index(self, unet: ToyUNet) -> None:
        path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0"
        parent, leaf = _get_parent(unet, path)
        assert leaf == "0"
        assert isinstance(parent, nn.ModuleList)


class TestSetSubmodule:
    def test_attribute_replacement(self, unet: ToyUNet) -> None:
        attn = _first_attn(unet)
        new_layer = nn.Linear(32, 32)
        _set_submodule(attn, "to_q", new_layer)
        assert attn.to_q is new_layer

    def test_modulelist_replacement(self, unet: ToyUNet) -> None:
        attn = _first_attn(unet)
        new_layer = nn.Linear(32, 32)
        _set_submodule(attn.to_out, "0", new_layer)
        assert attn.to_out[0] is new_layer
        # Sibling (Dropout at index 1) untouched
        assert isinstance(attn.to_out[1], nn.Dropout)


# ---------------------------------------------------------------------------
# Injection tests
# ---------------------------------------------------------------------------

class TestInject:
    def test_targets_become_lora(self, unet: ToyUNet) -> None:
        inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        attn = _first_attn(unet)
        assert isinstance(attn.to_q, LoRALinear)
        assert isinstance(attn.to_k, LoRALinear)
        assert isinstance(attn.to_v, LoRALinear)
        assert isinstance(attn.to_out[0], LoRALinear)
        assert isinstance(attn.add_q_proj, LoRALinear)
        # Dropout at to_out[1] must remain untouched
        assert isinstance(attn.to_out[1], nn.Dropout)

    def test_returns_only_lora_params(self, unet: ToyUNet) -> None:
        trainable = inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        # Each LoRALinear contributes 2 params (lora_A, lora_B).
        # 2 down_blocks × 1 attention × 1 transformer_block × 2 attn × 8 targets = 32 LoRALinears
        # → 64 trainable params total
        assert len(trainable) == 2 * 8 * 2 * 2

    def test_base_weights_frozen(self, unet: ToyUNet) -> None:
        inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        # Every non-LoRA param must have requires_grad=False
        for name, p in unet.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_starts_identical_to_base(self, unet: ToyUNet) -> None:
        """At injection, lora_B = 0 → output must equal pre-injection output."""
        x = torch.randn(2, 32)
        with torch.no_grad():
            y_before = unet(x)
        inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        with torch.no_grad():
            y_after = unet(x)
        assert torch.allclose(y_before, y_after, atol=1e-6)

    def test_grads_only_into_lora_after_step(self, unet: ToyUNet) -> None:
        trainable = inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        # Perturb so loss has nonzero grad w.r.t. lora_B (which starts at 0)
        with torch.no_grad():
            for p in trainable:
                if p.shape[1] == 4:  # lora_B has shape (out, rank)
                    p.normal_(std=0.1)
        x = torch.randn(2, 32, requires_grad=False)
        unet(x).sum().backward()
        # Some LoRA param must have a grad
        assert any(p.grad is not None for p in trainable)
        # No frozen base param should have a grad
        for name, p in unet.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                assert p.grad is None, f"{name} got a grad — should be frozen"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_safetensors(self, unet: ToyUNet, tmp_path: Path) -> None:
        inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        out = save_lora_weights(unet, tmp_path, rank=4, alpha=8.0, target_modules=DEFAULT_TARGET_MODULES)
        assert out.exists()
        assert out.name == "pytorch_lora_weights.safetensors"

    def test_save_keys_match_diffusers_convention(self, unet: ToyUNet, tmp_path: Path) -> None:
        inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        save_lora_weights(unet, tmp_path)
        from safetensors.torch import load_file
        state = load_file(str(tmp_path / "pytorch_lora_weights.safetensors"))
        # Every key starts with "unet." and ends with ".lora.down.weight" or ".lora.up.weight"
        for k in state:
            assert k.startswith("unet."), f"bad prefix: {k}"
            assert k.endswith(".lora.down.weight") or k.endswith(".lora.up.weight"), f"bad suffix: {k}"

    def test_roundtrip_preserves_weights(self, unet: ToyUNet, tmp_path: Path) -> None:
        trainable = inject_lora_into_unet(unet, rank=4, alpha=8.0, verbose=False)
        # Perturb LoRA weights so save isn't trivial (everything was zero/random)
        with torch.no_grad():
            for p in trainable:
                p.normal_(std=0.1)
        # Snapshot inputs/output BEFORE save so we can compare after reload
        x = torch.randn(2, 32)
        with torch.no_grad():
            y_before = unet(x)
        save_lora_weights(unet, tmp_path, rank=4, alpha=8.0, target_modules=DEFAULT_TARGET_MODULES)

        # Build a fresh UNet, load the weights, expect identical output
        torch.manual_seed(0)
        unet2 = ToyUNet(d=32)
        load_lora_weights(unet2, tmp_path)
        with torch.no_grad():
            y_after = unet2(x)
        assert torch.allclose(y_before, y_after, atol=1e-6), \
            f"round-trip diverged: max abs diff = {(y_before - y_after).abs().max()}"
