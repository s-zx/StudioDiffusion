"""
Adapters package.

LoRA exports (lightweight: torch + safetensors only) are always available.
`IPAdapterSDXL` requires `diffusers` + `transformers`; imported lazily so the
package surface still works in lightweight envs (e.g. `tests/test_imports`
on a fresh CPU venv).
"""

from .lora import (
    DEFAULT_TARGET_MODULES,
    LoRALinear,
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

try:
    from .ip_adapter.model import IPAdapterSDXL  # noqa: F401

    __all__.append("IPAdapterSDXL")
except ImportError:  # diffusers / transformers not installed
    IPAdapterSDXL = None  # type: ignore[assignment]

