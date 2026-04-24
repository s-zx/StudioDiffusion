"""
CLIP embedding diversity metrics for generated image sets.

Measures whether a generated batch has collapsed into a narrow visual mode.
Higher pairwise cosine distance implies more diverse outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image


class CLIPDiversity:
    """Compute diversity statistics from normalized CLIP image embeddings."""

    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        self.model = model.to(device).eval()
        self.preprocess = preprocess

    @torch.no_grad()
    def _embed_images(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        all_feats = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i : i + batch_size]:
                try:
                    batch.append(self.preprocess(Image.open(p).convert("RGB")))
                except Exception:
                    continue
            if not batch:
                continue
            inp = torch.stack(batch).to(self.device)
            feats = self.model.encode_image(inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 768))

    def score(self, images: list[Path]) -> dict[str, float]:
        embs = self._embed_images(images)
        n = len(embs)
        if n < 2:
            return {
                "clip_diversity_mean_pairwise_distance": 0.0,
                "clip_diversity_mean_pairwise_similarity": 1.0 if n == 1 else 0.0,
                "clip_diversity_mean_nearest_neighbor_similarity": 1.0 if n == 1 else 0.0,
                "clip_diversity_sample_count": float(n),
            }

        sims = embs @ embs.T
        upper = np.triu_indices(n, k=1)
        pairwise_sims = sims[upper]
        mean_pairwise_similarity = float(pairwise_sims.mean())
        mean_pairwise_distance = float((1.0 - pairwise_sims).mean())

        nn_sims = sims.copy()
        np.fill_diagonal(nn_sims, -np.inf)
        mean_nn_similarity = float(nn_sims.max(axis=1).mean())
        return {
            "clip_diversity_mean_pairwise_distance": mean_pairwise_distance,
            "clip_diversity_mean_pairwise_similarity": mean_pairwise_similarity,
            "clip_diversity_mean_nearest_neighbor_similarity": mean_nn_similarity,
            "clip_diversity_sample_count": float(n),
        }
