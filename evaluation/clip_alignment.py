"""
Platform aesthetic alignment via CLIP embeddings.

Primary evaluation metric. Measures whether generated images land in the
correct platform's distribution using:
  1. Mean cosine similarity to held-out real platform images.
  2. k-NN classification accuracy (generated image → predicted platform).

Usage
-----
from evaluation import CLIPPlatformAlignment
evaluator = CLIPPlatformAlignment(device="cuda")
evaluator.build_reference_embeddings(platform_image_dirs)
results = evaluator.evaluate(generated_images, target_platform="shopify")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


Platform = Literal["shopify", "etsy", "ebay"]
PLATFORMS: list[str] = ["shopify", "etsy", "ebay"]


class CLIPPlatformAlignment:
    """CLIP-based platform cluster alignment evaluator."""

    MODEL_NAME = "ViT-L-14"
    PRETRAINED = "openai"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.ref_embeddings: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _embed_images(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        all_feats = []
        for i in tqdm(range(0, len(paths), batch_size), desc="CLIP embedding", leave=False):
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

    def build_reference_embeddings(
        self, platform_dirs: dict[str, Path]
    ) -> None:
        """
        Pre-compute mean embeddings for each platform's held-out real images.

        Parameters
        ----------
        platform_dirs : dict mapping platform name → directory of real images.
        """
        for platform, img_dir in platform_dirs.items():
            paths = sorted(
                p for p in Path(img_dir).rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            if not paths:
                raise ValueError(f"No images found in {img_dir}")
            embs = self._embed_images(paths)
            self.ref_embeddings[platform] = embs
            print(f"[{platform}] Reference embeddings: {embs.shape}")

    def cosine_similarity_to_platform(
        self, gen_images: list[Path | np.ndarray], target_platform: str
    ) -> dict[str, float]:
        """
        Returns
        -------
        dict with keys:
            mean_cosine_sim      : mean cosine similarity to target platform centroid
            mean_cosine_sim_<p>  : mean cosine similarity to each platform centroid
        """
        if target_platform not in self.ref_embeddings:
            raise ValueError(f"Reference embeddings not built for '{target_platform}'.")

        paths = [p for p in gen_images if isinstance(p, Path)]
        gen_embs = self._embed_images(paths) if paths else np.empty((0, 768))

        results: dict[str, float] = {}
        target_centroid = self.ref_embeddings[target_platform].mean(axis=0, keepdims=True)
        target_centroid = target_centroid / (np.linalg.norm(target_centroid) + 1e-8)
        results["mean_cosine_sim"] = float((gen_embs @ target_centroid.T).mean())

        for platform, ref_embs in self.ref_embeddings.items():
            centroid = ref_embs.mean(axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            results[f"mean_cosine_sim_{platform}"] = float((gen_embs @ centroid.T).mean())

        return results

    def knn_platform_accuracy(
        self,
        gen_images: list[Path],
        target_platform: str,
        k: int = 5,
    ) -> float:
        """
        Train a k-NN on real platform images, then predict platform for each
        generated image. Returns classification accuracy (label = target_platform).
        """
        X_train, y_train = [], []
        for platform, embs in self.ref_embeddings.items():
            X_train.append(embs)
            y_train.extend([platform] * len(embs))
        X_train_arr = np.concatenate(X_train, axis=0)

        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train_arr, y_train)

        gen_embs = self._embed_images(gen_images)
        preds = knn.predict(gen_embs)
        labels = [target_platform] * len(preds)
        return float(accuracy_score(labels, preds))

    def evaluate(
        self,
        gen_images: list[Path],
        target_platform: str,
    ) -> dict[str, float]:
        """Run full CLIP alignment evaluation."""
        results = self.cosine_similarity_to_platform(gen_images, target_platform)
        results["knn_accuracy"] = self.knn_platform_accuracy(gen_images, target_platform)
        return results
