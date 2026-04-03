"""
LoRA adapter for SDXL UNet.

Injects low-rank matrices (A, B) into the attention projection layers of the
SDXL UNet using HuggingFace PEFT. Only rank-decomposed matrices are trained;
the base UNet remains frozen.

Reference: Hu et al. (2022) https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    ])
    dropout: float = 0.0


class LoRASDXL(nn.Module):
    """
    Wraps an SDXL UNet with PEFT LoRA adapters.

    Trainable parameters: low-rank A and B matrices inserted into every
    target attention module. Parameter count scales as:
        2 × rank × (d_in + d_out)  per target layer.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        lora_cfg: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        if lora_cfg is None:
            lora_cfg = LoRAConfig()

        peft_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias="none",
        )
        self.unet: PeftModel = get_peft_model(unet, peft_config)
        self.lora_cfg = lora_cfg

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def print_trainable_summary(self) -> None:
        total = sum(p.numel() for p in self.unet.parameters())
        trainable = self.num_trainable_parameters()
        print(
            f"LoRA trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(str(save_directory))

    @classmethod
    def load_pretrained(
        cls,
        unet: UNet2DConditionModel,
        load_directory: str | Path,
        lora_cfg: Optional[LoRAConfig] = None,
    ) -> "LoRASDXL":
        instance = cls.__new__(cls)
        super(LoRASDXL, instance).__init__()
        instance.lora_cfg = lora_cfg or LoRAConfig()
        instance.unet = PeftModel.from_pretrained(unet, str(load_directory))
        return instance

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)
