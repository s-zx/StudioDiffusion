"""
Product boundary preservation evaluation.

Measures whether the generated image preserves the original product's
foreground region using:
  1. Round-trip IoU: re-extract SAM2 mask from generated image, compare to
     original mask.
  2. LPIPS in the product region: perceptual similarity within the foreground.

Usage
-----
from evaluation import BoundaryPreservation
evaluator = BoundaryPreservation(device="cuda")
results = evaluator.evaluate(original_image, generated_image, original_mask)
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union + 1e-8)


class BoundaryPreservation:
    """
    Evaluates product boundary preservation via round-trip IoU and LPIPS.

    Parameters
    ----------
    sam2_extractor : SAM2Extractor instance for re-extracting masks.
    device : torch device string.
    """

    def __init__(self, sam2_extractor=None, device: str = "cuda") -> None:
        self.sam2 = sam2_extractor
        self.device = device

        if _LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

    def round_trip_iou(
        self,
        generated: np.ndarray | Image.Image,
        original_mask: np.ndarray,
    ) -> float:
        """
        Re-extracts a mask from `generated` using SAM2, then computes IoU
        against the original mask.
        """
        if self.sam2 is None:
            raise RuntimeError("SAM2Extractor required for round-trip IoU. Pass sam2_extractor=.")
        if isinstance(generated, Image.Image):
            generated = np.array(generated.convert("RGB"))
        gen_mask = self.sam2.extract(generated)
        return _iou(gen_mask, original_mask.astype(bool))

    @torch.no_grad()
    def lpips_in_mask(
        self,
        original: np.ndarray | Image.Image,
        generated: np.ndarray | Image.Image,
        mask: np.ndarray,
    ) -> float:
        """
        Perceptual similarity (LPIPS) computed only within the foreground mask region.
        Lower = more similar (better preservation).
        """
        if self.lpips_fn is None:
            raise ImportError("lpips not installed. Run: pip install lpips")

        if isinstance(original, Image.Image):
            original = np.array(original.convert("RGB"))
        if isinstance(generated, Image.Image):
            generated = np.array(generated.convert("RGB"))

        h, w = mask.shape
        original = np.array(Image.fromarray(original).resize((w, h)))
        generated = np.array(Image.fromarray(generated).resize((w, h)))

        # Zero out background
        mask_3c = mask[:, :, None].astype(np.float32)
        orig_masked = (original.astype(np.float32) * mask_3c).clip(0, 255).astype(np.uint8)
        gen_masked  = (generated.astype(np.float32) * mask_3c).clip(0, 255).astype(np.uint8)

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(img).float() / 127.5 - 1.0  # [-1, 1]
            return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        score = self.lpips_fn(to_tensor(orig_masked), to_tensor(gen_masked))
        return float(score.squeeze().cpu())

    def evaluate(
        self,
        original: np.ndarray | Image.Image,
        generated: np.ndarray | Image.Image,
        original_mask: np.ndarray,
    ) -> dict[str, float]:
        """
        Returns
        -------
        dict with keys: round_trip_iou, lpips_in_mask
        """
        results: dict[str, float] = {}

        if self.sam2 is not None:
            results["round_trip_iou"] = self.round_trip_iou(generated, original_mask)

        if self.lpips_fn is not None:
            results["lpips_in_mask"] = self.lpips_in_mask(original, generated, original_mask)

        return results
