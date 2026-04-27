"""
Multi-dimensional aesthetic scoring.

Uses LAION Aesthetic Predictor v2 (MLP head on CLIP embeddings) to score
overall aesthetic quality, and exposes composition, lighting, and color
harmony sub-scores via CLIP zero-shot comparison against curated prompts.

Usage
-----
from evaluation import AestheticScorer

# Sub-dimension scores only (no checkpoint required):
scorer = AestheticScorer(checkpoint=None, device="cpu")
scores = scorer.score_detailed(image)   # {"composition": 0.7, "lighting": 0.6, "color": 0.8}
c = scorer.score_composition(image)
l = scorer.score_lighting(image)
k = scorer.score_color(image)

# Full scoring including LAION overall quality (checkpoint required):
scorer = AestheticScorer(device="cuda")
q = scorer.score(image)                 # float, higher = more aesthetic
qs = scorer.score_batch(images)         # list[float]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    LAION Aesthetic Predictor v2 + CLIP zero-shot sub-dimension scoring.

    Weights: https://github.com/christophschuhmann/improved-aesthetic-predictor
    Place at checkpoints/aesthetic_predictor_v2.pth

    Pass checkpoint=None to use sub-dimension scoring only (no predictor weights needed).
    """

    CHECKPOINT = "checkpoints/aesthetic_predictor_v2.pth"
    CLIP_MODEL = "ViT-L-14"
    CLIP_PRETRAINED = "openai"

    # Positive / negative prompt pairs per sub-dimension.
    # Multiple prompts per pole are averaged to a single embedding for robustness.
    _DIMENSION_PROMPTS: dict[str, dict[str, list[str]]] = {
        "composition": {
            "positive": [
                "well-composed product photo with centered subject and balanced framing",
                "product photography with excellent composition, rule of thirds, clean background",
            ],
            "negative": [
                "poorly framed product photo with subject cut off or crowded edges",
                "bad composition, off-center, unbalanced, product barely visible",
            ],
        },
        "lighting": {
            "positive": [
                "evenly lit product photography with soft natural lighting and subtle shadows",
                "professional studio lighting, well-exposed, no harsh shadows or overexposure",
            ],
            "negative": [
                "harshly lit product photo with blown highlights and deep black shadows",
                "dark underexposed image, harsh flash shadows, poor uneven lighting",
            ],
        },
        "color": {
            "positive": [
                "harmonious color palette with consistent tones in product photography",
                "beautiful color composition, balanced hues, vibrant yet cohesive colors",
            ],
            "negative": [
                "clashing colors, muddy discolored tones, inconsistent color palette",
                "oversaturated garish colors or washed out desaturated product photo",
            ],
        },
    }

    def __init__(
        self,
        checkpoint: str | Path | None = CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        self.device = device

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            self.CLIP_MODEL, pretrained=self.CLIP_PRETRAINED
        )
        self.clip_model = clip_model.to(device).eval()
        self.preprocess = preprocess

        # Aesthetic predictor MLP — optional
        if checkpoint is not None:
            self.mlp = _AestheticMLP(input_size=768).to(device)
            ckpt = torch.load(str(checkpoint), map_location=device)
            self.mlp.load_state_dict(ckpt)
            self.mlp.eval()
        else:
            self.mlp = None

        # Pre-compute and cache text embeddings for sub-dimension scoring
        tokenizer = open_clip.get_tokenizer(self.CLIP_MODEL)
        self._dim_embeddings: dict[str, dict[str, torch.Tensor]] = {}
        for dim, prompts in self._DIMENSION_PROMPTS.items():
            self._dim_embeddings[dim] = {
                "positive": self._encode_texts(prompts["positive"], tokenizer),
                "negative": self._encode_texts(prompts["negative"], tokenizer),
            }

    @torch.no_grad()
    def _encode_texts(self, texts: list[str], tokenizer) -> torch.Tensor:
        """Encode a list of texts, mean-pool, and return a unit (1, d) tensor."""
        tokens = tokenizer(texts).to(self.device)
        feats = self.clip_model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
        mean = feats.mean(dim=0, keepdim=True)
        return F.normalize(mean, dim=-1)

    @torch.no_grad()
    def _encode_image(self, image: Image.Image | np.ndarray | str | Path) -> torch.Tensor:
        """Return a unit (1, d) CLIP image embedding."""
        inp = self.preprocess(self._coerce_image(image)).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(inp)
        return F.normalize(feat.float(), dim=-1)

    @staticmethod
    def _coerce_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.convert("RGB")

    @staticmethod
    def _compute_subscore(
        img_feat: torch.Tensor,  # (1, d), unit vector
        pos_emb: torch.Tensor,   # (1, d), unit vector
        neg_emb: torch.Tensor,   # (1, d), unit vector
    ) -> float:
        """
        Softmax-normalized similarity: P(image matches positive concept).
        Returns a float in [0, 1]; 0.5 when positive == negative.
        """
        logits = torch.cat(
            [(img_feat * pos_emb).sum(dim=-1, keepdim=True),
             (img_feat * neg_emb).sum(dim=-1, keepdim=True)],
            dim=-1,
        )  # (1, 2)
        prob = torch.softmax(logits, dim=-1)
        return float(prob[0, 0].cpu())

    def _subscore_for_dim(self, image: Image.Image | np.ndarray | str | Path, dim: str) -> float:
        feat = self._encode_image(image)
        return self._compute_subscore(
            feat,
            self._dim_embeddings[dim]["positive"],
            self._dim_embeddings[dim]["negative"],
        )

    def score_composition(self, image: Image.Image | np.ndarray | str | Path) -> float:
        """Composition quality score in [0, 1] (higher = better)."""
        return self._subscore_for_dim(image, "composition")

    def score_lighting(self, image: Image.Image | np.ndarray | str | Path) -> float:
        """Lighting quality score in [0, 1] (higher = better)."""
        return self._subscore_for_dim(image, "lighting")

    def score_color(self, image: Image.Image | np.ndarray | str | Path) -> float:
        """Color harmony score in [0, 1] (higher = better)."""
        return self._subscore_for_dim(image, "color")

    def score_detailed(
        self, image: Image.Image | np.ndarray | str | Path
    ) -> dict[str, float]:
        """
        Return all three sub-dimension scores in one forward pass.
        The image is encoded once and reused for all three dimensions.
        """
        feat = self._encode_image(image)
        return {
            dim: self._compute_subscore(
                feat,
                self._dim_embeddings[dim]["positive"],
                self._dim_embeddings[dim]["negative"],
            )
            for dim in ("composition", "lighting", "color")
        }

    @torch.no_grad()
    def score_batch_detailed(
        self,
        images: list[Image.Image | np.ndarray | str | Path],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """
        Sub-dimension scores for a list of images.
        Images are CLIP-encoded in batches; text embeddings are reused across all images.
        Returns a list of {"composition": float, "lighting": float, "color": float}.
        """
        all_results: list[dict[str, float]] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = [
                self.preprocess(self._coerce_image(img))
                for img in images[i : i + batch_size]
            ]
            inp = torch.stack(batch_imgs).to(self.device)
            feats = self.clip_model.encode_image(inp)
            feats = F.normalize(feats.float(), dim=-1)  # (B, d)
            for feat in feats:
                feat = feat.unsqueeze(0)  # (1, d)
                all_results.append({
                    dim: self._compute_subscore(
                        feat,
                        self._dim_embeddings[dim]["positive"],
                        self._dim_embeddings[dim]["negative"],
                    )
                    for dim in ("composition", "lighting", "color")
                })
        return all_results

    def _require_mlp(self) -> None:
        if self.mlp is None:
            raise RuntimeError(
                "Overall aesthetic score requires a checkpoint. "
                "Pass checkpoint= to AestheticScorer or use score_detailed() "
                "for sub-dimension scoring without a checkpoint."
            )

    @torch.no_grad()
    def score(self, image: Image.Image | np.ndarray | str | Path) -> float:
        """Overall LAION aesthetic score (float). Requires checkpoint."""
        self._require_mlp()
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
        """Overall LAION aesthetic score for a list of images. Requires checkpoint."""
        self._require_mlp()
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
