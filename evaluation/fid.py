"""
Fréchet Inception Distance (FID) for generated image sets.

This implementation uses torchvision's Inception v3 pooled features and keeps
the API intentionally small for project evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


_INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class FIDScorer:
    """Compute FID between a generated set and a real reference set."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights, aux_logits=False, transform_input=False)
        self.model.fc = nn.Identity()
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def _embed_images(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for path in paths[i : i + batch_size]:
                try:
                    batch.append(_INCEPTION_TRANSFORM(Image.open(path).convert("RGB")))
                except Exception:
                    continue
            if not batch:
                continue
            inp = torch.stack(batch).to(self.device)
            out = self.model(inp)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.empty((0, 2048))

    @staticmethod
    def _covariance(arr: np.ndarray) -> np.ndarray:
        if len(arr) < 2:
            raise ValueError("FID requires at least two images in each set.")
        return np.cov(arr, rowvar=False)

    @staticmethod
    def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.clip(vals, a_min=0.0, a_max=None)
        return (vecs * np.sqrt(vals)) @ vecs.T

    def score(self, generated_images: list[Path], real_images: list[Path]) -> float:
        gen_feats = self._embed_images(generated_images)
        real_feats = self._embed_images(real_images)
        if len(gen_feats) < 2 or len(real_feats) < 2:
            raise ValueError("FID requires at least two readable generated and real images.")

        mu_gen = gen_feats.mean(axis=0)
        mu_real = real_feats.mean(axis=0)
        sigma_gen = self._covariance(gen_feats)
        sigma_real = self._covariance(real_feats)

        diff = mu_gen - mu_real
        covmean = self._matrix_sqrt_psd(sigma_gen @ sigma_real)
        fid = diff @ diff + np.trace(sigma_gen + sigma_real - 2.0 * covmean)
        return float(np.real(fid))
