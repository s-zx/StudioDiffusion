"""
Multi-dimensional aesthetic scoring.

Uses LAION Aesthetic Predictor v2 (MLP head on CLIP embeddings) to score
overall aesthetic quality, and exposes hooks for composition, lighting, and
color harmony sub-scores via separate lightweight predictors.

Usage
-----
from evaluation import AestheticScorer
scorer = AestheticScorer(device="cuda")
score = scorer.score(image)            # float, higher = more aesthetic
scores = scorer.score_batch(images)   # list[float]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import open_clip
from PIL import Image


class _AestheticMLP(nn.Module):
    """LAION Aesthetic Predictor v2 MLP head (fits on top of CLIP ViT-L/14 embeddings)."""

    def __init__(self, input_size: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScorer:
    """
    LAION Aesthetic Predictor v2.

    Weights: https://github.com/christophschuhmann/improved-aesthetic-predictor
    Place at checkpoints/aesthetic_predictor_v2.pth
    """

    CHECKPOINT = "checkpoints/aesthetic_predictor_v2.pth"
    CLIP_MODEL = "ViT-L-14"
    CLIP_PRETRAINED = "openai"

    def __init__(
        self,
        checkpoint: str | Path = CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        self.device = device

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            self.CLIP_MODEL, pretrained=self.CLIP_PRETRAINED
        )
        self.clip_model = clip_model.to(device).eval()
        self.preprocess = preprocess

        self.mlp = _AestheticMLP(input_size=768).to(device)
        ckpt = torch.load(str(checkpoint), map_location=device)
        self.mlp.load_state_dict(ckpt)
        self.mlp.eval()

    @staticmethod
    def _coerce_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.convert("RGB")

    @torch.no_grad()
    def score(self, image: Image.Image | np.ndarray | str | Path) -> float:
        inp = self.preprocess(self._coerce_image(image)).unsqueeze(0).to(self.device)
        clip_feat = self.clip_model.encode_image(inp)
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
        score = self.mlp(clip_feat.float())
        return float(score.squeeze().cpu())

    @torch.no_grad()
    def score_batch(
        self,
        images: list[Image.Image | np.ndarray | str | Path],
        batch_size: int = 32,
    ) -> list[float]:
        all_scores = []
        for i in range(0, len(images), batch_size):
            batch_imgs = []
            for img in images[i : i + batch_size]:
                batch_imgs.append(self.preprocess(self._coerce_image(img)))
            inp = torch.stack(batch_imgs).to(self.device)
            clip_feat = self.clip_model.encode_image(inp)
            clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
            scores = self.mlp(clip_feat.float()).squeeze(-1)
            all_scores.extend(scores.cpu().tolist())
        return all_scores
