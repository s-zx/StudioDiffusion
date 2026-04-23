"""
Minimal inference smoke for trained IP-Adapter checkpoints.

Generates one image per platform using the trained IP-Adapter + SDXL base
(no ControlNet, no segmentation — simplest useful signal to confirm the
adapter produces stylistically different outputs for the three platforms).

Why a separate script from inference/generate.py:
  - generate.py uses diffusers' pipe.load_ip_adapter(...) which expects
    the H94-style single-file checkpoint format. Our IPAdapterSDXL
    checkpoint is two files (image_proj_model.pt + ip_attn_processors.pt),
    so we load it via our own IPAdapterSDXL.load_pretrained classmethod.
  - generate.py also chains ControlNet-canny with a SAM2 binary mask,
    which is a semantic mismatch. Smoke skips ControlNet entirely.
  - generate.py hardcodes device="cuda". Smoke auto-detects mps.

Usage:
    python inference/smoke.py                    # all 3 platforms, default reference
    python inference/smoke.py --platform shopify
    python inference/smoke.py --reference path/to/your/product.jpg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image
from torchvision import transforms

from adapters.ip_adapter.model import IPAdapterSDXL


SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"
CLIP_IMAGE_ENCODER = "openai/clip-vit-large-patch14-336"

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


def pick_default_reference() -> Path:
    """Pick the first image in shopify's val manifest as a shared reference
    across all three adapter runs (keeps content constant, varies adapter)."""
    manifest = Path("data/platform_sets/manifests/shopify_val.csv")
    platform_dir = Path("data/platform_sets/shopify")
    with open(manifest, newline="", encoding="utf-8") as f:
        first_row = next(csv.DictReader(f))
    src = Path(first_row["image_path"])
    for candidate in [
        platform_dir / f"{src.parent.name}_{src.name}",
        platform_dir / src.name,
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve reference image from {manifest}")


def generate_for_platform(
    platform: str,
    reference_path: Path,
    output_path: Path,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    adapter_ckpt = Path(f"checkpoints/ip_adapter/{platform}/final")
    if not adapter_ckpt.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_ckpt}")

    print(f"[{platform}] loading pipeline ({dtype}, {device})…")
    vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE, vae=vae, torch_dtype=dtype,
    ).to(device)

    print(f"[{platform}] loading adapter from {adapter_ckpt}…")
    adapter = IPAdapterSDXL.load_pretrained(
        unet=pipe.unet,
        load_directory=str(adapter_ckpt),
        image_encoder_id=CLIP_IMAGE_ENCODER,
        num_tokens=16,
        adapter_scale=1.0,
    )
    # Cast every adapter submodule (image_proj_model, to_k_ip/to_v_ip,
    # image_encoder) to the pipeline's dtype+device. adapter.unet is the
    # same object as pipe.unet; the cast is idempotent for it.
    adapter = adapter.to(device=device, dtype=dtype)

    # Encode reference image → IP prompt embeddings (cond + uncond for CFG)
    print(f"[{platform}] encoding reference: {reference_path.name}")
    ref_image = Image.open(reference_path).convert("RGB")
    clip_input = CLIP_TRANSFORM(ref_image).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        cond_ip, uncond_ip = adapter.encode_image(clip_input)
    # CFG: pipe duplicates latents [uncond, cond]; match the same order here
    ip_hidden_states = torch.cat([uncond_ip, cond_ip], dim=0)

    print(f"[{platform}] generating {height}x{width}, {num_inference_steps} steps…")
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        cross_attention_kwargs={"ip_hidden_states": ip_hidden_states},
        generator=generator,
    ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    print(f"[{platform}] saved → {output_path}")

    # Free VRAM before the next platform (keep CLIP weights cached on disk)
    del pipe, adapter, vae
    if device == "mps":
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", choices=["shopify", "etsy", "ebay"],
        help="Run a single platform. Omit to run all three.",
    )
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Reference product image. Defaults to the first entry in shopify_val.csv.",
    )
    parser.add_argument("--prompt", default="a professional product photograph")
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted, artifacts, watermark, text",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--device", default=None, help="mps / cpu / cuda. Auto-detected if unset.")
    parser.add_argument(
        "--dtype", default="fp16", choices=["fp16", "fp32"],
        help="fp16 is faster; fall back to fp32 if MPS complains.",
    )
    args = parser.parse_args()

    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    reference = args.reference or pick_default_reference()
    platforms = [args.platform] if args.platform else ["shopify", "etsy", "ebay"]

    print(f"device={device}  dtype={dtype}  reference={reference}")
    print(f"prompt={args.prompt!r}\n")

    for p in platforms:
        out = args.output_dir / f"{p}_{reference.stem}_seed{args.seed}.png"
        generate_for_platform(
            platform=p,
            reference_path=reference,
            output_path=out,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        print()


if __name__ == "__main__":
    main()
