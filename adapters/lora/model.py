"""
LoRA injection / save / load for an SDXL UNet.

This module turns an off-the-shelf `UNet2DConditionModel` into a LoRA-tuned UNet
by surgically replacing selected `nn.Linear` layers with `LoRALinear` wrappers,
then provides helpers to persist/restore just the LoRA deltas in a format that
`diffusers`' `pipe.load_lora_weights(...)` can read.

Public surface
--------------
- `inject_lora_into_unet(unet, ...) -> list[nn.Parameter]`
- `save_lora_weights(unet, save_directory) -> None`
- `load_lora_weights(unet, load_directory, ...) -> None`

References
----------
- microsoft/LoRA  (algorithm)
- huggingface/peft  (wrap-not-subclass module-replacement pattern)
- huggingface/diffusers  (on-disk key naming: `lora.down.weight` / `lora.up.weight`)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

from .layers import LoRALinear

# SDXL attention projections we adapt by default.
# - Self-attn (attn1) and cross-attn (attn2): to_q / to_k / to_v / to_out.0
# - SDXL's *added* cross-attention (text-time conditioning): add_q_proj / add_k_proj / add_v_proj / to_add_out
# Dropping the "add_*" entries silently breaks SDXL.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_parent(root: nn.Module, dotted_path: str) -> tuple[nn.Module, str]:
    """
    Resolve `root.<dotted_path>` to (parent_module, leaf_attr_name).

    Example:
        path = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        returns (root.down_blocks[0]...attn1, "to_q")
    """
    parts = dotted_path.split('.')
    current_module = root
    for part in parts[:-1]:
        current_module = getattr(current_module, part)
    return current_module, parts[-1]


def _set_submodule(parent: nn.Module, leaf: str, new_child: nn.Module) -> None:
    """
    Replace a child of `parent` named `leaf` with `new_child`.

    `nn.ModuleList` / `nn.Sequential` use integer indices; everything else uses
    attribute access. Using `setattr` works for both because PyTorch overrides
    `__setattr__` on ModuleList/Sequential to accept stringified ints — but it
    silently coerces the type, so we branch explicitly to be safe.
    """
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = new_child  
    else:
        setattr(parent, leaf, new_child)  


def _name_matches_target(module_name: str, targets: Iterable[str]) -> bool:
    """
    True iff `module_name` ends in any of `targets`.

    A target like "to_out.0" must match the *suffix* of the dotted path, not
    just the leaf — that's what distinguishes `attn.to_out.0` (the Linear) from
    `attn.to_out.1` (the Dropout). Matching the leaf "0" alone would be wrong.
    """
    return any(module_name == t or module_name.endswith("." + t) for t in targets)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_lora_into_unet(
    unet: nn.Module,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    verbose: bool = True,
) -> list[nn.Parameter]:
    """
    Walk `unet.named_modules()` and replace every `nn.Linear` whose dotted path
    ends in one of `target_modules` with a `LoRALinear` wrapping it.

    After injection:
      - All non-LoRA UNet params have `requires_grad = False`.
      - The returned list contains *only* the trainable LoRA params
        (suitable for passing straight to `torch.optim.AdamW`).

    Parameters
    ----------
    unet : nn.Module
        Typically a `diffusers.UNet2DConditionModel`. Modified in place.
    target_modules : iterable of str
        Module-name suffixes to match (see `DEFAULT_TARGET_MODULES`).
    rank, alpha, dropout : LoRA hyperparameters forwarded to `LoRALinear`.
    verbose : bool
        If True, print a one-line summary of trainable / total params.

    Returns
    -------
    list[nn.Parameter]
        The LoRA `lora_A` / `lora_B` parameters, in injection order.
    """
    unet.requires_grad_(False)
    module_snapshot=list(unet.named_modules())
    trainable_params: list[nn.Parameter] = []
    for (name,module) in module_snapshot:
        if _name_matches_target(name, target_modules) and isinstance(module, nn.Linear):
            parent, leaf = _get_parent(unet, name)
            new = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            _set_submodule(parent, leaf, new)
            trainable_params.extend([new.lora_A, new.lora_B])
    if verbose:
        n_train = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in unet.parameters())
        print(f"[LoRA] trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
    return trainable_params


def save_lora_weights(
    unet: nn.Module,
    save_directory: str | Path,
    *,
    rank: int | None = None,
    alpha: float | None = None,
    target_modules: Iterable[str] | None = None,
) -> Path:
    """
    Serialize just the LoRA deltas in a layout `pipe.load_lora_weights()` can read.

    On-disk layout
    --------------
    `<save_directory>/pytorch_lora_weights.safetensors` with keys:
        unet.<dotted module path>.lora.down.weight   <- lora_A
        unet.<dotted module path>.lora.up.weight     <- lora_B

    e.g. `unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight`

    Plus `<save_directory>/lora_config.json` recording rank / alpha /
    target_modules so `load_lora_weights` doesn't need them passed in again
    (single source of truth — see plan §"`load_lora_weights` re-injection").

    Parameters
    ----------
    unet : nn.Module
        A UNet that was passed through `inject_lora_into_unet`.
    save_directory : path-like
        Created if missing.
    rank, alpha, target_modules :
        Optional metadata for `lora_config.json`. If None we just omit them
        from the json (the safetensors file is still self-describing by shape).

    Returns
    -------
    Path
        Path to the written `.safetensors` file.
    """
    save_dir=Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    state_dict={}
    for (name, module) in unet.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[f"unet.{name}.lora.down.weight"] = module.lora_A.detach().cpu()
            state_dict[f"unet.{name}.lora.up.weight"] = module.lora_B.detach().cpu()
    safetensors_save(state_dict, str(save_dir / "pytorch_lora_weights.safetensors"))

    if rank is not None or alpha is not None or target_modules is not None:
        cfg = {}
        if rank is not None: cfg["rank"] = rank
        if alpha is not None: cfg["alpha"] = alpha
        if target_modules is not None: cfg["target_modules"] = list(target_modules)
        (save_dir / "lora_config.json").write_text(json.dumps(cfg, indent=2))

    return save_dir / "pytorch_lora_weights.safetensors"


def load_lora_weights(
    unet: nn.Module,
    load_directory: str | Path,
    *,
    rank: int | None = None,
    alpha: float | None = None,
    dropout: float = 0.0,
    target_modules: Iterable[str] | None = None,
) -> list[nn.Parameter]:
    """
    Inverse of `save_lora_weights`. Re-injects LoRA into `unet` (using
    `lora_config.json` if present, else the kwargs), then loads the deltas.

    Returns the list of trainable LoRA params (same contract as
    `inject_lora_into_unet`) so callers can resume training if they want.
    """
    load_dir = Path(load_directory)
    cfg_path = load_dir / "lora_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            if rank is None: rank = cfg.get("rank", rank)
            if alpha is None: alpha = cfg.get("alpha", alpha)
            if target_modules is None: target_modules = cfg.get("target_modules", target_modules)

    # Fall back to defaults if neither kwargs nor JSON provided values.
    if rank is None: rank = 16
    if alpha is None: alpha = 16.0
    if target_modules is None: target_modules = DEFAULT_TARGET_MODULES

    trainable_params = inject_lora_into_unet(
        unet, target_modules=target_modules, rank=rank, alpha=alpha, dropout=dropout, verbose=False
    )
    state = safetensors_load(str(load_dir / "pytorch_lora_weights.safetensors"))

    modules_by_name = {name: m for name, m in unet.named_modules() if isinstance(m, LoRALinear)}

    for key, tensor in state.items():
        if key.startswith("unet.") and key.endswith(".lora.down.weight"):
            module_path = key[len("unet."): -len(".lora.down.weight")]
            modules_by_name[module_path].lora_A.data.copy_(tensor)
        elif key.startswith("unet.") and key.endswith(".lora.up.weight"):
            module_path = key[len("unet."): -len(".lora.up.weight")]
            modules_by_name[module_path].lora_B.data.copy_(tensor)
        else:
            raise ValueError(f"Unexpected key format: {key}")
    return trainable_params