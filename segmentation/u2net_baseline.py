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

import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, dirate: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _RSU(nn.Module):
    """Residual U-block used by U²Net."""

    def __init__(self, name: str, height: int, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.rebnconvin = _conv_bn_relu(in_ch, out_ch)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.encoders.append(_conv_bn_relu(out_ch, mid_ch))
        for i in range(height - 2):
            self.encoders.append(_conv_bn_relu(mid_ch, mid_ch))
        self.encoders.append(_conv_bn_relu(mid_ch, mid_ch, dirate=2))

        for _ in range(height - 1):
            self.decoders.append(_conv_bn_relu(mid_ch * 2, mid_ch))
        self.decoders.append(_conv_bn_relu(mid_ch * 2, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        hx_in = self.rebnconvin(hx)
        encoder_outs = [hx_in]
        hx = hx_in
        for i, enc in enumerate(self.encoders[:-1]):
            hx = enc(hx)
            encoder_outs.append(hx)
            hx = self.pool(hx)
        hx = self.encoders[-1](hx)
        for dec, skip in zip(self.decoders, reversed(encoder_outs)):
            hx = F.interpolate(hx, size=skip.shape[2:], mode="bilinear", align_corners=False)
            hx = dec(torch.cat([hx, skip], dim=1))
        return hx + hx_in


class U2Net(nn.Module):
    """Lightweight U²Net for binary salient object detection."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = _RSU("En_1", 7, 3, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = _RSU("En_2", 6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = _RSU("En_3", 5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = _RSU("En_4", 4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = _RSU("En_5", 4, 512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = _RSU("En_6", 4, 512, 256, 512)

        self.stage5d = _RSU("De_5", 4, 1024, 256, 512)
        self.stage4d = _RSU("De_4", 4, 1024, 128, 256)
        self.stage3d = _RSU("De_3", 5, 512, 64, 128)
        self.stage2d = _RSU("De_2", 6, 256, 32, 64)
        self.stage1d = _RSU("De_1", 7, 128, 16, 64)

        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[2:]
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[2:], mode="bilinear", align_corners=False)

        hx5d = self.stage5d(torch.cat([up(hx6, hx5), hx5], dim=1))
        hx4d = self.stage4d(torch.cat([up(hx5d, hx4), hx4], dim=1))
        hx3d = self.stage3d(torch.cat([up(hx4d, hx3), hx3], dim=1))
        hx2d = self.stage2d(torch.cat([up(hx3d, hx2), hx2], dim=1))
        hx1d = self.stage1d(torch.cat([up(hx2d, hx1), hx1], dim=1))

        d1 = self.side1(hx1d)
        d2 = up(self.side2(hx2d), hx1d)
        d3 = up(self.side3(hx3d), hx1d)
        d4 = up(self.side4(hx4d), hx1d)
        d5 = up(self.side5(hx5d), hx1d)
        d6 = up(self.side6(hx6), hx1d)
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))
        return torch.sigmoid(d0), [torch.sigmoid(s) for s in [d1, d2, d3, d4, d5, d6]]


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
        self.model = U2Net().to(device)
        state = torch.load(str(checkpoint), map_location=device)
        self.model.load_state_dict(state)
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
        d0, _ = self.model(inp)
        prob = d0.squeeze().cpu().numpy()
        prob_resized = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize(orig_size, Image.BILINEAR)
        ) / 255.0
        return prob_resized >= self.threshold
