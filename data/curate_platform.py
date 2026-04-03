"""
Platform-specific dataset curation.

Uses CLIP embeddings to score candidate images against per-platform reference
archetype prompts, then selects the top-K images per platform. Outputs a
directory of symlinked (or copied) images ready for adapter training.

Usage
-----
python data/curate_platform.py \
    --source data/raw/abo/images/small \
    --output data/platform_sets \
    --n_per_platform 400 \
    --device cuda
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm


PLATFORM_PROMPTS: dict[str, list[str]] = {
    "shopify": [
        "product photo with clean white background, studio lighting, minimal props",
        "e-commerce product image, high contrast, neutral background, sharp focus",
    ],
    "etsy": [
        "handmade product photo with warm lighting, natural textures, lifestyle props",
        "artisanal product photography, cozy background, earthy tones, soft shadows",
    ],
    "ebay": [
        "product listing photo, bright even lighting, plain background, utilitarian clarity",
        "used item photograph, straightforward lighting, neutral background, clear details",
    ],
}


def load_clip(device: str) -> tuple:
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
) -> tuple[torch.Tensor, list[Path]]:
    all_feats, valid_paths = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding images"):
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
    return torch.cat(all_feats, dim=0), valid_paths


def curate(
    source: Path,
    output: Path,
    n_per_platform: int,
    device: str,
) -> None:
    image_paths = sorted(
        p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"Found {len(image_paths)} candidate images in {source}")

    model, preprocess, tokenizer = load_clip(device)
    image_feats, valid_paths = embed_images(image_paths, model, preprocess, device)

    for platform, prompts in PLATFORM_PROMPTS.items():
        text_feats = embed_texts(prompts, model, tokenizer, device)
        # Average over all archetype prompts
        archetype = text_feats.mean(dim=0, keepdim=True).to(image_feats.device)
        scores = (image_feats @ archetype.T).squeeze(-1)

        top_indices = scores.topk(min(n_per_platform, len(valid_paths))).indices
        out_dir = output / platform
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx in top_indices:
            src = valid_paths[idx]
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

        print(f"[{platform}] Selected {len(top_indices)} images → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/platform_sets"))
    parser.add_argument("--n_per_platform", type=int, default=400)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    curate(args.source, args.output, args.n_per_platform, args.device)


if __name__ == "__main__":
    main()
