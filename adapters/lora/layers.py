"""
LoRA layer — from-scratch implementation.

Reference: Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685

Core idea
---------
For a frozen linear layer h = W0 @ x, LoRA adds a low-rank update:

    h = W0 @ x  +  (alpha / r) * B @ A @ x

where A ∈ R^{r × d_in}, B ∈ R^{d_out × r}, and r << min(d_in, d_out).
Only A and B are trainable; W0 stays frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Wraps a frozen `nn.Linear` and adds a trainable low-rank update.

    Forward: y = base(x) + scaling * (x @ A.T @ B.T)
             where scaling = alpha / rank.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # TODO: validate rank > 0
        if rank <= 0:
            raise ValueError(f"Invalid rank {rank}: must be > 0")

        # TODO: store base, rank, alpha, and compute self.scaling = alpha / rank
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # TODO: pull in_features / out_features from base
        in_features = base.in_features
        out_features = base.out_features

        # TODO: create trainable low-rank matrices
        #   lora_A: shape (rank, in_features), init with kaiming_uniform_(a=math.sqrt(5))
        #   lora_B: shape (out_features, rank), init to zeros
        if rank > 0: 
            self.lora_A=nn.Parameter(torch.empty(rank,in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B= nn.Parameter(torch.zeros(out_features,rank))
        
        for p in self.base.parameters():
            p.requires_grad_(False)

        # TODO: dropout on the LoRA branch input
        #   if dropout > 0: nn.Dropout(p=dropout) else nn.Identity()
        self.lora_dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # TODO: freeze every parameter in self.base (requires_grad_(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = base(x) + scaling * LoRA(x)

        Uses F.linear (= x @ W.T) twice so the LoRA branch broadcasts cleanly
        over arbitrary leading batch/sequence dims of x.
        """
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )
