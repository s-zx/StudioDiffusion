"""
Smoke test: verify imports resolve without errors.
Actual model tests require GPU + downloaded checkpoints.

LoRA package surface is being rebuilt — see docs/lora-implementation-plan.md.
This test currently asserts only what File 1 (`LoRALinear`) provides; extend
it as Files 2 (`inject_lora_into_unet`, save/load) and 3 (`train`) land.
"""

import importlib
import importlib.util


def test_adapter_imports():
    """Lightweight imports — must work with just torch installed."""
    from adapters import LoRALinear  # noqa: F401
    from adapters.lora import LoRALinear as _LoRALinear  # noqa: F401


def test_optional_heavy_imports():
    """Skip cleanly if any heavy runtime dep isn't installed in this env."""
    import pytest

    required = ("diffusers", "transformers", "torchvision", "accelerate", "omegaconf")
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"missing heavy deps: {', '.join(missing)}")

    from segmentation import SAM2Extractor, U2NetExtractor, evaluate_masks  # noqa: F401
    from adapters import IPAdapterSDXL  # noqa: F401
    from evaluation import (  # noqa: F401
        AestheticScorer,
        BoundaryPreservation,
        CLIPPlatformAlignment,
        DINOv2Fidelity,
    )
    from inference import generate_product_image  # noqa: F401
