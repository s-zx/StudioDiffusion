"""
Download ABO images using the correct image_id → path mapping from images.csv.gz.

Flow:
  1. Download images/metadata/images.csv.gz  (6 MB)
  2. Download listings/metadata/*.json.gz    (already done if present)
  3. Join image_id → path, filter US, sample balanced set
  4. Download sample images with 16 threads
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

BUCKET = "amazon-berkeley-objects"
REGION = "us-east-1"


def get_s3():
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED),
    )


def download_bytes(s3, key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


# ---------------------------------------------------------------------------
# Step 1 — Build image_id → S3 path mapping
# ---------------------------------------------------------------------------

def build_image_map(s3, local_cache: Path) -> dict[str, str]:
    """Return {image_id: 'prefix/filename.jpg'} from images/metadata/images.csv.gz."""
    if local_cache.exists():
        print(f"[ABO] Loading image map from cache: {local_cache}")
    else:
        print("[ABO] Downloading images/metadata/images.csv.gz (6 MB)...")
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        data = download_bytes(s3, "images/metadata/images.csv.gz")
        local_cache.write_bytes(data)

    mapping: dict[str, str] = {}
    with gzip.open(local_cache, "rt", encoding="utf-8") as f:
        header = f.readline()  # image_id,height,width,path
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                mapping[parts[0]] = parts[3]  # image_id → path

    print(f"[ABO] Image map: {len(mapping):,} entries")
    return mapping


# ---------------------------------------------------------------------------
# Step 2 — Download metadata (if not already present)
# ---------------------------------------------------------------------------

def ensure_metadata(s3, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="listings/metadata/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json.gz"):
                keys.append(obj["Key"])

    print(f"[ABO] {len(keys)} metadata files on S3")
    downloaded = []
    for key in tqdm(keys, desc="Metadata files"):
        local = dest / Path(key).name
        if not local.exists():
            data = download_bytes(s3, key)
            local.write_bytes(data)
        downloaded.append(local)
    print(f"[ABO] Metadata ready in {dest}")
    return downloaded


# ---------------------------------------------------------------------------
# Step 3 — Parse metadata + sample
# ---------------------------------------------------------------------------

def parse_and_sample(
    meta_files: list[Path],
    image_map: dict[str, str],
    n_total: int,
    country: str = "US",
    seed: int = 42,
) -> list[dict]:
    by_cat: dict[str, list] = defaultdict(list)
    for gz in tqdm(meta_files, desc="Parsing metadata"):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if country and item.get("country") != country:
                    continue
                img_id = item.get("main_image_id", "")
                if not img_id or img_id not in image_map:
                    continue
                pt = item.get("product_type", "UNKNOWN")
                if isinstance(pt, list):
                    pt = pt[0].get("value", "UNKNOWN") if pt else "UNKNOWN"
                by_cat[pt].append({
                    "item_id":      item.get("item_id", ""),
                    "image_id":     img_id,
                    "s3_path":      image_map[img_id],
                    "product_type": pt,
                })

    total_available = sum(len(v) for v in by_cat.values())
    print(f"[ABO] {total_available:,} US items with resolvable images across {len(by_cat)} categories")

    # Balanced sample
    cats = sorted(by_cat.keys())
    n_per_cat = max(1, n_total // len(cats))
    rng = random.Random(seed)
    sampled = []
    for cat in cats:
        pool = by_cat[cat]
        sampled.extend(rng.sample(pool, min(n_per_cat, len(pool))))
    rng.shuffle(sampled)
    result = sampled[:n_total]
    print(f"[ABO] Sampled {len(result):,} items across {len({r['product_type'] for r in result})} categories")
    return result


# ---------------------------------------------------------------------------
# Step 4 — Download images
# ---------------------------------------------------------------------------

def _download_one(s3, s3_path: str, images_dir: Path) -> tuple[str, bool]:
    key = f"images/small/{s3_path}"
    local_path = images_dir / s3_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return s3_path, True
    try:
        s3.download_file(BUCKET, key, str(local_path))
        return s3_path, True
    except Exception as e:
        return s3_path, False


def download_images(records: list[dict], images_dir: Path, max_workers: int = 16) -> int:
    s3 = get_s3()
    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, s3, r["s3_path"], images_dir): r
            for r in records
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            _, ok = fut.result()
            if ok:
                success += 1
    print(f"[ABO] Downloaded {success:,} / {len(records):,} images → {images_dir}")
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images",     type=int,  default=3000)
    parser.add_argument("--metadata_dir", type=Path, default=Path("data/raw/abo/metadata"))
    parser.add_argument("--images_dir",   type=Path, default=Path("data/raw/abo/images/small"))
    parser.add_argument("--country",      type=str,  default="US")
    parser.add_argument("--max_workers",  type=int,  default=16)
    args = parser.parse_args()

    s3 = get_s3()

    # Image ID → S3 path mapping
    image_map = build_image_map(
        s3, local_cache=Path("data/raw/abo/metadata/images.csv.gz")
    )

    # Metadata
    meta_files = ensure_metadata(s3, args.metadata_dir)

    # Sample
    sample = parse_and_sample(meta_files, image_map, args.n_images, args.country)
    if not sample:
        print("[ABO] No records found.", file=sys.stderr)
        sys.exit(1)

    # Download
    download_images(sample, args.images_dir, args.max_workers)
    print("[ABO] Done. Next: python data/preprocess_abo.py")


if __name__ == "__main__":
    main()
