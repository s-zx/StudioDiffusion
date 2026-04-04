"""
Platform-specific dataset curation.

Uses CLIP embeddings to score candidate images against per-platform reference
archetype prompts, then assigns each image to the platform it matches best
(hard assignment — no cross-platform duplicates). Outputs:
  - Copied images under data/platform_sets/{platform}/
  - Train/val split manifests under data/platform_sets/manifests/

Changes from v1
---------------
- --sources accepts multiple directories (ABO + DeepFashion2 + optional extras)
- Hard platform assignment: every image goes to exactly one platform
- Cross-platform deduplication is automatic (argmax assignment)
- Train/val 80/20 split written as CSV manifests for each platform
- Progress saved to an embedding cache so re-runs are fast

Usage
-----
# Basic (ABO only)
python data/curate_platform.py \
    --sources data/raw/abo/images/small \
    --n_per_platform 400

# Multi-source (ABO + DeepFashion2)
python data/curate_platform.py \
    --sources data/raw/abo/images/small data/raw/deepfashion2/train/image \
    --n_per_platform 400 \
    --output data/platform_sets \
    --device cuda

# With ABO manifest to carry product_type through to the split CSV
python data/curate_platform.py \
    --sources data/raw/abo/images/small data/raw/deepfashion2/train/image \
    --abo_manifest data/processed/abo_manifest.csv \
    --df2_manifest data/processed/deepfashion2_manifest.csv \
    --n_per_platform 400 \
    --device cuda
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Platform archetype prompts
# ---------------------------------------------------------------------------

PLATFORM_PROMPTS: dict[str, list[str]] = {
    "shopify": [
        "product photo with clean white background, studio lighting, minimal props",
        "e-commerce product image, high contrast, neutral background, sharp focus",
        "commercial product photography, clean backdrop, professional lighting",
    ],
    "etsy": [
        "handmade product photo with warm lighting, natural textures, lifestyle props",
        "artisanal product photography, cozy background, earthy tones, soft shadows",
        "craft item photo, rustic background, warm color temperature, natural light",
    ],
    "ebay": [
        "product listing photo, bright even lighting, plain background, utilitarian clarity",
        "used item photograph, straightforward lighting, neutral background, clear details",
        "resale product photo, high clarity, minimal styling, white or grey background",
    ],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def load_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def embed_texts(
    prompts: list[str],
    model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    feats = model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_images(
    paths: list[Path],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
    cache_path: Path | None = None,
) -> tuple[torch.Tensor, list[Path]]:
    """
    Embed a list of images with CLIP. If cache_path is given, load/save a
    .pt cache of (feats, valid_path_strings) so repeated runs are fast.
    """
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        print(f"[CLIP] Loaded embedding cache ({len(cached['paths'])} images) from {cache_path}")
        return cached["feats"], [Path(p) for p in cached["paths"]]

    all_feats: list[torch.Tensor] = []
    valid_paths: list[Path] = []

    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP-embedding images"):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
                valid_paths.append(p)
            except Exception:
                continue
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())

    all_feats_tensor = torch.cat(all_feats, dim=0) if all_feats else torch.empty(0, 768)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feats": all_feats_tensor, "paths": [str(p) for p in valid_paths]}, cache_path)
        print(f"[CLIP] Embedding cache saved → {cache_path}")

    return all_feats_tensor, valid_paths


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def build_path_metadata(
    valid_paths: list[Path],
    abo_manifest: pd.DataFrame | None,
    df2_manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build a DataFrame indexed by image path with optional category metadata
    carried over from the ABO / DF2 manifests.
    """
    df = pd.DataFrame({"image_path": [str(p) for p in valid_paths]})

    # Merge ABO product_type
    if abo_manifest is not None and "image_path" in abo_manifest.columns:
        abo_sub = abo_manifest[["image_path", "product_type"]].rename(
            columns={"product_type": "category"}
        )
        df = df.merge(abo_sub, on="image_path", how="left")
    else:
        df["category"] = ""

    # Fill in DeepFashion2 category where ABO didn't match
    if df2_manifest is not None and "image_path" in df2_manifest.columns:
        df2_sub = df2_manifest[["image_path", "category_name"]].rename(
            columns={"category_name": "category_df2"}
        )
        df = df.merge(df2_sub, on="image_path", how="left")
        df["category"] = df["category"].fillna(df.get("category_df2", ""))
        if "category_df2" in df.columns:
            df.drop(columns=["category_df2"], inplace=True)

    df["category"] = df["category"].fillna("").astype(str)
    return df


def write_split_manifests(
    selected_paths: list[Path],
    meta_df: pd.DataFrame,
    platform: str,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    """Write {platform}_train.csv and {platform}_val.csv to output_dir/manifests/."""
    str_paths = [str(p) for p in selected_paths]
    sub = meta_df[meta_df["image_path"].isin(str_paths)].copy()

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sub))
    n_val = max(1, int(len(sub) * val_fraction))
    val_idx  = idx[:n_val]
    train_idx = idx[n_val:]

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    sub.iloc[train_idx].to_csv(manifests_dir / f"{platform}_train.csv", index=False)
    sub.iloc[val_idx].to_csv(manifests_dir / f"{platform}_val.csv",   index=False)

    print(
        f"[{platform}] Manifests written: "
        f"{len(train_idx)} train, {len(val_idx)} val  →  {manifests_dir}/"
    )


# ---------------------------------------------------------------------------
# Main curation logic
# ---------------------------------------------------------------------------

def collect_image_paths(sources: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for src in sources:
        found = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        print(f"[sources] {src}: {len(found):,} images")
        paths.extend(found)
    return paths


def curate(
    sources: list[Path],
    output: Path,
    n_per_platform: int,
    device: str,
    abo_manifest_path: Path | None,
    df2_manifest_path: Path | None,
    val_fraction: float,
    seed: int,
    embedding_cache: Path | None,
) -> None:
    # ---- Gather all candidate images ----
    image_paths = collect_image_paths(sources)
    print(f"\n[curate] Total candidate images: {len(image_paths):,}")

    if not image_paths:
        raise ValueError("No images found in the provided --sources directories.")

    # ---- CLIP embeddings ----
    model, preprocess, tokenizer = load_clip(device)
    image_feats, valid_paths = embed_images(
        image_paths, model, preprocess, device, cache_path=embedding_cache
    )
    print(f"[curate] Embedded {len(valid_paths):,} images")

    # ---- Per-platform archetype embeddings ----
    platforms = list(PLATFORM_PROMPTS.keys())
    archetype_feats = torch.stack([
        embed_texts(PLATFORM_PROMPTS[p], model, tokenizer, device).mean(0)
        for p in platforms
    ])  # shape: (n_platforms, embed_dim)
    archetype_feats = archetype_feats / archetype_feats.norm(dim=-1, keepdim=True)
    archetype_feats = archetype_feats.to(image_feats.device)

    # ---- Score matrix: (n_images, n_platforms) ----
    scores = (image_feats @ archetype_feats.T)  # (N, 3)

    # ---- Hard assignment: each image → best-matching platform only ----
    # This prevents the same image appearing in two platform sets.
    best_platform_idx = scores.argmax(dim=1).numpy()  # (N,)
    scores_np = scores.numpy()                         # (N, 3)

    # ---- Load metadata manifests if provided ----
    abo_manifest  = pd.read_csv(abo_manifest_path)  if abo_manifest_path  and abo_manifest_path.exists()  else None
    df2_manifest  = pd.read_csv(df2_manifest_path)  if df2_manifest_path  and df2_manifest_path.exists()  else None
    meta_df = build_path_metadata(valid_paths, abo_manifest, df2_manifest)

    # ---- Per-platform selection and copy ----
    for p_idx, platform in enumerate(platforms):
        # Images assigned to this platform, sorted by their score for this platform
        assigned_mask = best_platform_idx == p_idx
        assigned_indices = np.where(assigned_mask)[0]

        # Sort by score descending and take top-N
        assigned_scores = scores_np[assigned_indices, p_idx]
        sorted_order = np.argsort(-assigned_scores)
        top_indices = assigned_indices[sorted_order[:n_per_platform]]

        out_dir = output / platform
        out_dir.mkdir(parents=True, exist_ok=True)

        selected_paths: list[Path] = []
        for idx in top_indices:
            src = valid_paths[idx]
            dst = out_dir / src.name
            # Handle filename collisions from multiple sources
            if dst.exists() and dst.resolve() != src.resolve():
                dst = out_dir / f"{src.parent.name}_{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)
            selected_paths.append(dst)

        print(
            f"[{platform}] Assigned {len(assigned_indices):,} images, "
            f"selected top {len(top_indices)}  →  {out_dir}"
        )

        # ---- Train / val split manifests ----
        # Re-map selected paths back to their original source paths for metadata lookup
        source_selected = [valid_paths[i] for i in top_indices]
        write_split_manifests(
            source_selected, meta_df, platform, output, val_fraction, seed
        )

    print(f"\n[curate] Platform sets written to {output}/")
    print("[curate] Train/val manifests written to", output / "manifests/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate platform-specific image sets using CLIP similarity."
    )
    parser.add_argument(
        "--sources", type=Path, nargs="+", required=True,
        help="One or more directories to scan for candidate images "
             "(e.g. data/raw/abo/images/small data/raw/deepfashion2/train/image)."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/platform_sets"),
        help="Root output directory. Per-platform subdirs are created automatically."
    )
    parser.add_argument(
        "--n_per_platform", type=int, default=400,
        help="Maximum number of images to select per platform."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--abo_manifest", type=Path, default=None,
        help="Path to data/processed/abo_manifest.csv (for category metadata in manifests)."
    )
    parser.add_argument(
        "--df2_manifest", type=Path, default=None,
        help="Path to data/processed/deepfashion2_manifest.csv."
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.2,
        help="Fraction of each platform set held out for validation (default: 0.2)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible train/val splits."
    )
    parser.add_argument(
        "--embedding_cache", type=Path, default=Path("data/processed/clip_embedding_cache.pt"),
        help="Path to save/load CLIP embedding cache (speeds up re-runs). "
             "Pass '' to disable caching."
    )
    args = parser.parse_args()

    cache = args.embedding_cache if str(args.embedding_cache) else None

    curate(
        sources=args.sources,
        output=args.output,
        n_per_platform=args.n_per_platform,
        device=args.device,
        abo_manifest_path=args.abo_manifest,
        df2_manifest_path=args.df2_manifest,
        val_fraction=args.val_fraction,
        seed=args.seed,
        embedding_cache=cache,
    )


if __name__ == "__main__":
    main()
