"""
Segmentation evaluation: compare predicted masks to DeepFashion2 ground-truth
annotations using standard metrics.

Metrics
-------
- IoU (Intersection over Union)
- Dice coefficient
- Boundary F1 (BF score)
- Precision / Recall

Usage
-----
python segmentation/evaluate_masks.py \
    --gt_dir  data/raw/deepfashion2/masks \
    --pred_dir data/processed/masks/sam2 \
    --model_name sam2

python segmentation/evaluate_masks.py \
    --gt_dir  data/raw/deepfashion2/masks \
    --pred_dir data/processed/masks/u2net \
    --model_name u2net
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union + 1e-8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    return 2.0 * float(inter) / float(pred.sum() + gt.sum() + 1e-8)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, dilation_px: int = 3) -> float:
    """Boundary F1: fraction of predicted / GT boundary pixels within dilation_px."""
    pred_b = pred ^ binary_dilation(pred, iterations=dilation_px)
    gt_b   = gt   ^ binary_dilation(gt,   iterations=dilation_px)
    prec = (pred_b & binary_dilation(gt_b, iterations=dilation_px)).sum() / (pred_b.sum() + 1e-8)
    rec  = (gt_b & binary_dilation(pred_b, iterations=dilation_px)).sum() / (gt_b.sum() + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return float(f1)


def load_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img) > 127


def evaluate_masks(
    gt_dir: Path,
    pred_dir: Path,
) -> dict[str, float]:
    """
    Compare all predicted masks in pred_dir against ground-truth masks in gt_dir.
    Masks are matched by filename stem.

    Returns
    -------
    dict with keys: mean_iou, mean_dice, mean_bf1, n_samples
    """
    gt_paths = {p.stem: p for p in gt_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg"}}
    pred_paths = {p.stem: p for p in pred_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg"}}

    common_stems = sorted(set(gt_paths) & set(pred_paths))
    if not common_stems:
        raise ValueError(f"No matching mask files found between {gt_dir} and {pred_dir}")

    ious, dices, bf1s = [], [], []
    for stem in tqdm(common_stems, desc="Evaluating"):
        gt_mask   = load_mask(gt_paths[stem])
        pred_mask = load_mask(pred_paths[stem])

        # Resize pred to GT size if needed
        if gt_mask.shape != pred_mask.shape:
            pred_pil = Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
            )
            pred_mask = np.array(pred_pil) > 127

        ious.append(iou(pred_mask, gt_mask))
        dices.append(dice(pred_mask, gt_mask))
        bf1s.append(boundary_f1(pred_mask, gt_mask))

    results = {
        "mean_iou":  float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_bf1":  float(np.mean(bf1s)),
        "n_samples": len(common_stems),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir",   type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--output", type=Path, default=Path("results/segmentation"))
    args = parser.parse_args()

    results = evaluate_masks(args.gt_dir, args.pred_dir)
    print(f"\n=== {args.model_name} segmentation metrics ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / f"{args.model_name}_seg_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
