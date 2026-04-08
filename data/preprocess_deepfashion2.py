"""
Preprocess DeepFashion2 annotations.

Converts per-image JSON annotation files (polygon segmentations) into binary
PNG mask files and produces a manifest CSV that the segmentation evaluation
pipeline (segmentation/evaluate_masks.py) and platform curation pipeline
(data/curate_platform.py) can both consume.

DeepFashion2 annotation format (per image JSON)
------------------------------------------------
{
  "source": "shop" | "user",
  "pair_id": int,
  "item1": {
    "category_name": str,       # e.g. "short sleeve top"
    "category_id":   int,
    "style":         int,
    "bounding_box":  [x1, y1, x2, y2],
    "landmarks":     [...],
    "segmentation":  [[x1, y1, x2, y2, ...]],   # list of polygons (flat coords)
    "scale":         int,
    "occlusion":     int,
    "zoom_in":       int,
    "viewpoint":     int
  },
  "item2": { ... }   # optional second item in the same image
}

Outputs
-------
data/processed/deepfashion2/masks/{split}/{image_id}.png
    Binary mask (255 = foreground, 0 = background).
    If the image contains multiple items, all segmentation polygons are merged.

data/processed/deepfashion2_manifest.csv
    Columns: image_path, mask_path, category_name, source, split

Usage
-----
# After running data/download_deepfashion2.sh
python data/preprocess_deepfashion2.py

# Custom paths
python data/preprocess_deepfashion2.py \
    --raw_dir  data/raw/deepfashion2 \
    --out_masks data/processed/deepfashion2/masks \
    --out_csv   data/processed/deepfashion2_manifest.csv \
    --splits    train validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


def rasterize_polygons(
    polygons: list[list[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert a list of flat-coordinate polygon lists to a binary uint8 mask.

    Parameters
    ----------
    polygons : list of flat [x1, y1, x2, y2, ...] coordinate lists
    width, height : canvas dimensions

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype uint8, values 0 or 255
    """
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if len(poly) < 6:
            continue
        # flat list → list of (x, y) tuples
        coords = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        if len(coords) >= 3:
            draw.polygon(coords, fill=255)
    return np.array(canvas)


def _get_image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) without fully decoding the image."""
    with Image.open(image_path) as img:
        return img.size  # (W, H)


def process_split(
    raw_dir: Path,
    split: str,
    out_masks_dir: Path,
    num_workers: int = 8,
) -> list[dict]:
    """
    Process one split (train / validation) using parallel workers.

    Returns a list of manifest row dicts.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    image_dir = raw_dir / split / "image"
    # DeepFashion2 uses 'annos' (not 'annots')
    annot_dir = raw_dir / split / "annos"

    if not annot_dir.exists():
        print(
            f"[DF2] WARNING: annotation directory not found: {annot_dir}\n"
            f"      Skipping split '{split}'.",
            file=sys.stderr,
        )
        return []

    annot_files = sorted(annot_dir.glob("*.json"))
    if not annot_files:
        print(f"[DF2] WARNING: no JSON files in {annot_dir}", file=sys.stderr)
        return []

    out_split_dir = out_masks_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DF2] {split}: {len(annot_files):,} annotations, {num_workers} workers")

    records = []
    errors = 0

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_process_one, ann_path, image_dir, out_split_dir): ann_path
            for ann_path in annot_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"[DF2] {split}", unit="img"):
            result = fut.result()
            if result is None:
                errors += 1
            else:
                records.append(result)

    print(f"[DF2] {split}: {len(records):,} masks saved, {errors:,} skipped")
    return records


def _process_one(
    ann_path: Path,
    image_dir: Path,
    out_split_dir: Path,
) -> dict | None:
    """Worker function: process a single annotation file (runs in subprocess)."""
    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    image_id   = ann_path.stem
    image_path = image_dir / f"{image_id}.jpg"
    if not image_path.exists():
        return None

    all_polygons: list[list[float]] = []
    category_names: list[str] = []

    for key in ("item1", "item2", "item3", "item4"):
        item = ann.get(key)
        if not item:
            continue
        seg = item.get("segmentation", [])
        if seg:
            all_polygons.extend(seg)
        cat = item.get("category_name", "")
        if cat:
            category_names.append(cat)

    if not all_polygons:
        return None

    try:
        w, h = _get_image_size(image_path)
    except Exception:
        return None

    mask = rasterize_polygons(all_polygons, w, h)
    mask_path = out_split_dir / f"{image_id}.png"
    Image.fromarray(mask).save(mask_path)

    return {
        "image_path":    str(image_path),
        "mask_path":     str(mask_path),
        "category_name": "|".join(category_names) if category_names else "unknown",
        "source":        ann.get("source", "unknown"),
        "split":         ann_path.parent.parent.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepFashion2 polygon annotations to binary mask PNGs."
    )
    parser.add_argument("--raw_dir",   type=Path, default=Path("data/raw/deepfashion2/deepfashion2_original_images"))
    parser.add_argument("--out_masks", type=Path, default=Path("data/processed/deepfashion2/masks"))
    parser.add_argument("--out_csv",   type=Path, default=Path("data/processed/deepfashion2_manifest.csv"))
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation"],
        help="Which splits to process."
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel worker processes for mask rasterization."
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(
            f"[DF2] ERROR: raw data directory not found: {args.raw_dir}\n"
            "      Run  bash data/download_deepfashion2.sh  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_records: list[dict] = []
    for split in args.splits:
        records = process_split(args.raw_dir, split, args.out_masks, num_workers=args.num_workers)
        all_records.extend(records)

    if not all_records:
        print("[DF2] ERROR: no records produced. Check your raw_dir and split names.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_records)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"\n[DF2] Manifest saved → {args.out_csv}  ({len(df):,} rows)")

    # Category distribution
    cat_dist = (
        df["category_name"]
        .str.split("|").explode()
        .value_counts()
        .head(15)
    )
    print("\n[DF2] Top 15 categories:")
    for cat, count in cat_dist.items():
        print(f"  {cat:<35s}  {count:>6,}")

    print("\n[DF2] Source distribution:")
    for src, count in df["source"].value_counts().items():
        print(f"  {src:<10s}  {count:>6,}")

    print("\nNext step: python data/curate_platform.py --sources ...")


if __name__ == "__main__":
    main()
