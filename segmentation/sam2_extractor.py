"""
SAM2 foreground mask extractor.

SAM2 is used as a frozen pretrained tool — no fine-tuning is performed.
Given a product image, it returns a binary foreground mask using the
largest connected segment (heuristic: product is the dominant object).

Install SAM2:
    pip install segment-anything-2
    # Download checkpoint, e.g.:
    # wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

Usage
-----
from segmentation import SAM2Extractor
extractor = SAM2Extractor(checkpoint="checkpoints/sam2_hiera_large.pt")
mask = extractor.extract(image)   # np.ndarray H×W bool
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False


class SAM2Extractor:
    """Zero-shot foreground mask extraction using Segment Anything 2."""

    DEFAULT_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
    DEFAULT_CONFIG = "sam2_hiera_l.yaml"

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        model_cfg: str = DEFAULT_CONFIG,
        device: str = "cuda",
    ) -> None:
        if not _SAM2_AVAILABLE:
            raise ImportError(
                "segment-anything-2 not installed. Run: pip install segment-anything-2"
            )
        sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = device

    def extract(
        self,
        image: np.ndarray | Image.Image,
        center_point: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Extract a binary foreground mask.

        Parameters
        ----------
        image:
            RGB image as numpy array (H, W, 3) or PIL Image.
        center_point:
            (x, y) pixel to use as the positive prompt point. Defaults to
            image center, which works well for centered product photos.

        Returns
        -------
        mask : np.ndarray
            Boolean mask of shape (H, W).
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        h, w = image.shape[:2]
        if center_point is None:
            center_point = (w // 2, h // 2)

        self.predictor.set_image(image)
        point_coords = np.array([[center_point[0], center_point[1]]])
        point_labels = np.array([1])  # foreground

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Pick highest-scoring mask
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)

    def extract_batch(
        self,
        images: list[np.ndarray | Image.Image],
    ) -> list[np.ndarray]:
        return [self.extract(img) for img in images]
