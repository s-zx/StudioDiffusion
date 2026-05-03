"""
U²Net baseline mask extractor.

Used as a segmentation baseline to compare against SAM2 on DeepFashion2
ground-truth annotations. U²Net is loaded from a local weights file.

Weights: https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view
Place at checkpoints/u2net.pth

Usage
-----
from segmentation import U2NetExtractor
extractor = U2NetExtractor(checkpoint="checkpoints/u2net.pth")
mask = extractor.extract(image)   # np.ndarray H×W bool
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Minimal U²Net architecture (full model defined inline to avoid submodule dep)
# ---------------------------------------------------------------------------


from segmentation_models.u2net.u2net_refactor import U2NET_full






# ---------------------------------------------------------------------------
# Public extractor class
# ---------------------------------------------------------------------------

_TRANSFORM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class U2NetExtractor:
    """Foreground mask extraction using a pretrained U²Net."""

    def __init__(
        self,
        checkpoint: str | Path = "checkpoints/u2net.pth",
        device: str = "cuda",
        threshold: float = 0.5,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self.model = U2NET_full().to(device)
        self.model.load_state_dict(torch.load(str(checkpoint), map_location=device))
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray or PIL.Image
            RGB image.

        Returns
        -------
        mask : np.ndarray bool, shape (H, W)
        """
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image)
        else:
            pil = image.convert("RGB")

        orig_size = pil.size  # (W, H)
        inp = _TRANSFORM(pil).unsqueeze(0).to(self.device)
        d0 = self.model(inp)[0]
        prob = d0.squeeze().cpu().numpy()
        prob_resized = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize(orig_size, Image.BILINEAR)
        ) / 255.0
        return prob_resized >= self.threshold
