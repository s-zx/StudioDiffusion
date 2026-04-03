"""
Generate BLIP-2 captions for training images.

Captions are saved as <image_stem>.txt alongside each image for use as
conditioning text during diffusion training.

Usage
-----
python data/generate_captions.py \
    --input data/platform_sets \
    --output data/processed/captions \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_blip2(device: str):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def caption_images(
    image_paths: list[Path],
    processor,
    model,
    device: str,
    batch_size: int = 8,
) -> dict[str, str]:
    results: dict[str, str] = {}
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Captioning"):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(None)

        valid = [(p, img) for p, img in zip(batch_paths, imgs) if img is not None]
        if not valid:
            continue
        paths_batch, imgs_batch = zip(*valid)

        inputs = processor(images=list(imgs_batch), return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for path, caption in zip(paths_batch, captions):
            results[str(path)] = caption.strip()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Root of platform_sets/")
    parser.add_argument("--output", type=Path, default=Path("data/processed/captions"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Found {len(image_paths)} images under {args.input}")

    processor, model = load_blip2(args.device)
    captions = caption_images(image_paths, processor, model, args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / "captions.json"
    with open(out_file, "w") as f:
        json.dump(captions, f, indent=2)

    # Also write per-image sidecar .txt files
    for path_str, caption in captions.items():
        p = Path(path_str)
        txt_path = args.output / p.parent.name / (p.stem + ".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(caption)

    print(f"Captions saved to {out_file}")


if __name__ == "__main__":
    main()
