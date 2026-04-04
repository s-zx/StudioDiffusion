"""
Preprocess Amazon Berkeley Objects (ABO) dataset.

Reads the gzipped JSON-lines metadata files, resolves image paths on disk,
and writes a clean manifest CSV for use by the platform curation pipeline.

Also handles rendered images: pairs each render PNG with its corresponding
segmentation mask PNG from the renders/segmentation_masks/ directory.

Outputs
-------
data/processed/abo_manifest.csv
    Columns: item_id, image_path, product_type, has_render, render_path, mask_path

Usage
-----
# After running data/download_abo.sh
python data/preprocess_abo.py

# Custom paths
python data/preprocess_abo.py \
    --metadata_dir data/raw/abo/metadata \
    --images_dir   data/raw/abo/images/small \
    --renders_dir  data/raw/abo/renders \
    --output       data/processed/abo_manifest.csv \
    --country      US
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ABO image filenames are stored as "<hash_prefix>/<image_id>.jpg"
# where hash_prefix is the first 2 characters of the image_id.
def _resolve_image_path(image_id: str, images_dir: Path) -> Path | None:
    prefix = image_id[:2]
    candidate = images_dir / prefix / f"{image_id}.jpg"
    return candidate if candidate.exists() else None


def _resolve_render_paths(
    item_id: str,
    renders_dir: Path,
) -> tuple[Path | None, Path | None]:
    """Return (render_png, mask_png) for spin-frame 0, or (None, None) if absent."""
    render_path = renders_dir / "png" / item_id / "0000.png"
    mask_path   = renders_dir / "segmentation_masks" / item_id / "0000.png"
    return (
        render_path if render_path.exists() else None,
        mask_path   if mask_path.exists()   else None,
    )


def parse_metadata(
    metadata_dir: Path,
    images_dir: Path,
    renders_dir: Path,
    country_filter: str | None,
) -> list[dict]:
    gz_files = sorted(metadata_dir.glob("*.json.gz"))
    if not gz_files:
        print(
            f"[ERROR] No .json.gz files found in {metadata_dir}.\n"
            "        Run  bash data/download_abo.sh  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    records = []
    skipped_country = 0
    skipped_no_image = 0

    for gz_path in tqdm(gz_files, desc="Parsing metadata files"):
        with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Country filter (ABO has US, JP, DE, UK, …)
                if country_filter and item.get("country") != country_filter:
                    skipped_country += 1
                    continue

                item_id      = item.get("item_id", "")
                main_img_id  = item.get("main_image_id", "")
                product_type = item.get("product_type", [{}])
                if isinstance(product_type, list) and product_type:
                    product_type = product_type[0].get("value", "UNKNOWN")
                else:
                    product_type = "UNKNOWN"

                image_path = _resolve_image_path(main_img_id, images_dir) if main_img_id else None
                if image_path is None:
                    skipped_no_image += 1
                    continue

                render_path, mask_path = _resolve_render_paths(item_id, renders_dir)

                records.append({
                    "item_id":      item_id,
                    "image_id":     main_img_id,
                    "image_path":   str(image_path),
                    "product_type": product_type,
                    "country":      item.get("country", ""),
                    "has_render":   render_path is not None,
                    "render_path":  str(render_path) if render_path else "",
                    "mask_path":    str(mask_path)   if mask_path   else "",
                })

    print(f"\n[ABO] Parsed {len(records):,} items")
    print(f"      Skipped (country filter): {skipped_country:,}")
    print(f"      Skipped (image not on disk): {skipped_no_image:,}")
    return records


def print_category_distribution(df: pd.DataFrame, top_n: int = 20) -> None:
    dist = df["product_type"].value_counts().head(top_n)
    print(f"\n[ABO] Top {top_n} product categories:")
    for cat, count in dist.items():
        bar = "#" * min(40, count // max(1, len(df) // 400))
        print(f"  {cat:<40s} {count:>6,}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ABO metadata into a manifest CSV.")
    parser.add_argument("--metadata_dir", type=Path, default=Path("data/raw/abo/metadata"))
    parser.add_argument("--images_dir",   type=Path, default=Path("data/raw/abo/images/small"))
    parser.add_argument("--renders_dir",  type=Path, default=Path("data/raw/abo/renders"))
    parser.add_argument("--output",       type=Path, default=Path("data/processed/abo_manifest.csv"))
    parser.add_argument(
        "--country", type=str, default="US",
        help="Filter to a single country code (US / JP / DE / UK / …). Pass '' to keep all."
    )
    args = parser.parse_args()

    country = args.country.strip() or None

    records = parse_metadata(
        metadata_dir=args.metadata_dir,
        images_dir=args.images_dir,
        renders_dir=args.renders_dir,
        country_filter=country,
    )

    df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\n[ABO] Manifest saved → {args.output}  ({len(df):,} rows)")
    print_category_distribution(df)

    renders_available = df["has_render"].sum()
    print(f"\n[ABO] Items with render + mask available: {renders_available:,} / {len(df):,}")
    print("\nNext step: python data/curate_platform.py --sources data/raw/abo/images/small ...")


if __name__ == "__main__":
    main()
