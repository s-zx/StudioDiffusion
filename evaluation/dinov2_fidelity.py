"""
Product identity fidelity via DINOv2 embeddings.

Measures how well the generated image preserves the original product's
geometry and texture, independent of background changes. Uses cosine
similarity between DINOv2 features of the original and generated product
regions (masked by SAM2 foreground mask).

Usage
-----
from evaluation import DINOv2Fidelity
scorer = DINOv2Fidelity(device="cuda")
score = scorer.score(original_image, generated_image, mask)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


_DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class DINOv2Fidelity:
    """Computes product identity fidelity using DINOv2 ViT-L/14."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        inp = _DINO_TRANSFORM(image).unsqueeze(0).to(self.device)
        feat = self.model(inp)
        return F.normalize(feat, dim=-1)

    def score(
        self,
        original: np.ndarray | Image.Image,
        generated: np.ndarray | Image.Image,
        mask: np.ndarray | None = None,
    ) -> float:
        """
        Parameters
        ----------
        original : RGB image of the original product.
        generated : RGB image of the generated output.
        mask : Optional binary foreground mask (H, W bool). If provided,
               background pixels are zeroed before embedding.

        Returns
        -------
        cosine_similarity : float in [-1, 1]. Higher = more identity-preserving.
        """
        if isinstance(original, np.ndarray):
            original = Image.fromarray(original)
        if isinstance(generated, np.ndarray):
            generated = Image.fromarray(generated)

        if mask is not None:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                original.size, Image.NEAREST
            )
            original = Image.composite(
                original, Image.new("RGB", original.size, (128, 128, 128)), mask_img
            )
            mask_gen = mask_img.resize(generated.size, Image.NEAREST)
            generated = Image.composite(
                generated, Image.new("RGB", generated.size, (128, 128, 128)), mask_gen
            )

        orig_feat = self._embed(original)
        gen_feat  = self._embed(generated)
        return float((orig_feat * gen_feat).sum().cpu())

    def score_batch(
        self,
        originals: list,
        generateds: list,
        masks: list | None = None,
    ) -> list[float]:
        if masks is None:
            masks = [None] * len(originals)
        return [
            self.score(o, g, m)
            for o, g, m in zip(originals, generateds, masks)
        ]
