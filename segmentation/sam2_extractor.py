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
    from segmentation_models.sam2.build_sam import build_sam2
    #from segmentation_models.sam2.sam2_image_predictor import SAM2ImagePredictor
    from segmentation_models.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
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
        self.predictor = SAM2AutomaticMaskGenerator(sam2_model, points_per_side=16,min_mask_region_area=500,pred_iou_thresh=0.8,
                                                    stability_score_thresh=0.95,
                                                    
                                                   )  
        self.device = device

    def extract(
        self,
        image: np.ndarray | Image.Image,
        
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

        

        mask=self.predictor.generate(image)
        
        if mask is None or len(mask) == 0:
            return np.zeros(image.shape[:2], dtype=bool)

        filtered = [m for m in mask if m['predicted_iou'] > 0.7]
        if filtered is None or len(filtered) == 0:
            scores=[i['predicted_iou'] for i in mask]
            best_idx = int(np.argmax(scores))
            return mask[best_idx]['segmentation'].astype(bool)
        areas=[i['area'] for i in filtered]

        best_idx = int(np.argmax(areas))
        segment=filtered[best_idx]['segmentation'].astype(bool)

        

        return segment

    def extract_batch(
        self,
        images: list[np.ndarray | Image.Image],
    ) -> list[np.ndarray]:
        return [self.extract(img) for img in images]
