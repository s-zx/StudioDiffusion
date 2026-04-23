"""
LoRA adapter package.

Public surface (per docs/lora-implementation-plan.md):
- File 1: LoRALinear                                                              ✅
- File 2: inject_lora_into_unet, save_lora_weights, load_lora_weights, DEFAULT_TARGET_MODULES  ✅
- File 3: train (CLI entry, see train.py)                                         🚧
"""

from .layers import LoRALinear
from .model import (
    DEFAULT_TARGET_MODULES,
    inject_lora_into_unet,
    load_lora_weights,
    save_lora_weights,
)

__all__ = [
    "LoRALinear",
    "DEFAULT_TARGET_MODULES",
    "inject_lora_into_unet",
    "save_lora_weights",
    "load_lora_weights",
]
