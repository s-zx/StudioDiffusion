"""
Multi-dimensional aesthetic scoring.

Uses LAION Aesthetic Predictor v2 (MLP head on CLIP embeddings) to score
overall aesthetic quality, and exposes composition, lighting, and color
harmony sub-scores via image statistics (no extra model weights needed).

Usage
-----
from evaluation import AestheticScorer

# Sub-dimension scores only (no model required):
scorer = AestheticScorer(checkpoint=None, device="cpu")
scores = scorer.score_detailed(image)   # {"composition": 0.7, "lighting": 0.6, "color": 0.8}
c = scorer.score_composition(image)
l = scorer.score_lighting(image)
k = scorer.score_color(image)
batch = scorer.score_batch_detailed(images)  # list[dict[str, float]]

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
    LAION Aesthetic Predictor v2 + image-statistics sub-dimension scoring.

    Weights: https://github.com/christophschuhmann/improved-aesthetic-predictor
    Place at checkpoints/aesthetic_predictor_v2.pth

    Pass checkpoint=None to use sub-dimension scoring only — no model downloads
    or GPU required; sub-scores are computed from pixel statistics.
    """

    CHECKPOINT = "checkpoints/aesthetic_predictor_v2.pth"
    CLIP_MODEL = "ViT-L-14"
    CLIP_PRETRAINED = "openai"

    def __init__(
        self,
        checkpoint: str | Path | None = CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.mlp = None
        self.clip_model = None
        self.preprocess = None

        if checkpoint is not None:
            import open_clip
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                self.CLIP_MODEL, pretrained=self.CLIP_PRETRAINED
            )
            self.clip_model = clip_model.to(device).eval()
            self.preprocess = preprocess

            self.mlp = _AestheticMLP(input_size=768).to(device)
            ckpt = torch.load(str(checkpoint), map_location=device)
            self.mlp.load_state_dict(ckpt)
            self.mlp.eval()

    # ------------------------------------------------------------------
    # Image statistics helpers (pure numpy — no model required)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(image: Image.Image | np.ndarray | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return np.array(image.convert("RGB"), dtype=np.float32)

    @staticmethod
    def _stat_composition(arr: np.ndarray) -> float:
        """
        Center-weight ratio: fraction of edge energy in the central 50×50% region.
        Well-composed product photos concentrate the subject toward the center.
        Returns float in [0, 1].
        """
        gray = arr.mean(axis=2)
        gy = np.abs(np.diff(gray, axis=0))   # (h-1, w)
        gx = np.abs(np.diff(gray, axis=1))   # (h, w-1)

        h, w = gray.shape
        grad = np.zeros((h, w), dtype=np.float32)
        grad[:h - 1, :] += gy
        grad[:, :w - 1] += gx

        total = grad.sum()
        if total < 1.0:
            return 0.5

        h0, h1 = h // 4, 3 * h // 4
        w0, w1 = w // 4, 3 * w // 4
        center_fraction = grad[h0:h1, w0:w1].sum() / total
        return float(np.clip(center_fraction, 0.0, 1.0))

    @staticmethod
    def _stat_lighting(arr: np.ndarray) -> float:
        """
        Lighting quality: combines exposure (mean brightness) and contrast (std).
        Ideal exposure centers near 128/255; ideal RMS contrast ~0.15.
        Returns float in [0, 1].
        """
        gray = arr.mean(axis=2) / 255.0          # [0, 1]
        mean_b = float(gray.mean())
        std_b = float(gray.std())

        # Gaussian peak at mid-exposure (0.5), drops for under/over-exposed images
        exposure = float(np.exp(-8.0 * (mean_b - 0.5) ** 2))

        # Gaussian peak at ideal contrast (~0.15 normalised std)
        contrast = float(np.exp(-20.0 * (std_b - 0.15) ** 2))

        return float(np.clip(0.5 * exposure + 0.5 * contrast, 0.0, 1.0))

    @staticmethod
    def _stat_color(arr: np.ndarray) -> float:
        """
        Hasler-Süsstrunk (2003) colorfulness metric, normalized to [0, 1].
        Higher = more colorful; gray/white images score near 0.
        Returns float in [0, 1].
        """
        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        raw = (
            float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
            + 0.3 * float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
        )
        # Normalize: raw ~0 (gray) to ~100+ (vivid); cap at 80 for [0, 1]
        return float(np.clip(raw / 80.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public sub-dimension API
    # ------------------------------------------------------------------

    def score_composition(self, image: Image.Image | np.ndarray | str | Path) -> float:
        """Composition quality score in [0, 1] (higher = subject more centered)."""
        return self._stat_composition(self._to_array(image))

    def score_lighting(self, image: Image.Image | np.ndarray | str | Path) -> float:
        """Lighting quality score in [0, 1] (higher = better exposure and contrast)."""
        return self._stat_lighting(self._to_array(image))

    def score_color(self, image: Image.Image | np.ndarray | str | Path) -> float:
        """Color richness score in [0, 1] (higher = more colorful)."""
        return self._stat_color(self._to_array(image))

    def score_detailed(
        self, image: Image.Image | np.ndarray | str | Path
    ) -> dict[str, float]:
        """
        All three sub-dimension scores in one call (array decoded once).
        Returns {"composition": float, "lighting": float, "color": float}.
        """
        arr = self._to_array(image)
        return {
            "composition": self._stat_composition(arr),
            "lighting":    self._stat_lighting(arr),
            "color":       self._stat_color(arr),
        }

    def score_batch_detailed(
        self,
        images: list[Image.Image | np.ndarray | str | Path],
    ) -> list[dict[str, float]]:
        """
        Sub-dimension scores for a list of images.
        Returns a list of {"composition": float, "lighting": float, "color": float}.
        """
        return [self.score_detailed(img) for img in images]

    # ------------------------------------------------------------------
    # Overall LAION aesthetic score (requires checkpoint + CLIP)
    # ------------------------------------------------------------------

    def _require_mlp(self) -> None:
        if self.mlp is None:
            raise RuntimeError(
                "Overall aesthetic score requires a checkpoint. "
                "Pass checkpoint= to AestheticScorer or use score_detailed() "
                "for sub-dimension scoring without a checkpoint."
            )

    @staticmethod
    def _coerce_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.convert("RGB")

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
