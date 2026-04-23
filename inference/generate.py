"""
End-to-end product image generation.

Pipeline:
  1. Extract foreground mask with SAM2.
  2. Load ControlNet + SDXL with trained IP-Adapter or LoRA.
  3. Generate platform-targeted image conditioned on mask + reference image.

Usage
-----
python inference/generate.py \
    --product   path/to/product.jpg \
    --platform  shopify \
    --adapter   ip_adapter \
    --adapter_ckpt checkpoints/ip_adapter/shopify/final \
    --prompt    "a product photo on a clean white background" \
    --output    outputs/shopify_result.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

from segmentation import SAM2Extractor


def mask_to_controlnet_conditioning(mask: np.ndarray) -> Image.Image:
    """Convert binary mask to a 3-channel conditioning image for ControlNet."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    rgb = np.stack([mask_uint8] * 3, axis=-1)
    return Image.fromarray(rgb)


def generate_product_image(
    product_path: str | Path,
    platform: str,
    adapter_type: str,
    adapter_ckpt: str | Path,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, artifacts",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    adapter_scale: float = 1.0,
    controlnet_conditioning_scale: float = 0.8,
    seed: int = 42,
    device: str = "cuda",
    output_path: str | Path | None = None,
) -> Image.Image:
    """
    Full pipeline: segment → condition → generate.

    Parameters
    ----------
    product_path : Path to the input product image.
    platform : One of 'shopify', 'etsy', 'ebay'.
    adapter_type : 'ip_adapter' or 'lora'.
    adapter_ckpt : Path to the trained adapter checkpoint directory.
    prompt : Text prompt for generation.
    output_path : If provided, saves the output image here.

    Returns
    -------
    PIL.Image of the generated product photo.
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    # ---- Step 1: Segment product ----
    extractor = SAM2Extractor(device=device)
    product_image = np.array(Image.open(product_path).convert("RGB"))
    mask = extractor.extract(product_image)
    control_image = mask_to_controlnet_conditioning(mask)

    # ---- Step 2: Load pipeline ----
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # ---- Step 3: Load adapter ----
    if adapter_type == "ip_adapter":
        pipe.load_ip_adapter(str(adapter_ckpt), subfolder="", weight_name="ip_adapter.bin")
        pipe.set_ip_adapter_scale(adapter_scale)
        ip_adapter_image = Image.open(product_path).convert("RGB")
    elif adapter_type == "lora":
        # Loads `pytorch_lora_weights.safetensors` written by
        # `adapters.lora.model.save_lora_weights` (diffusers attn-procs format).
        pipe.load_lora_weights(str(adapter_ckpt))
        ip_adapter_image = None
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    # ---- Step 4: Generate ----
    generate_kwargs: dict = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if ip_adapter_image is not None:
        generate_kwargs["ip_adapter_image"] = ip_adapter_image

    result = pipe(**generate_kwargs).images[0]

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_path))
        print(f"Saved to {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product",     type=Path, required=True)
    parser.add_argument("--platform",    type=str,  required=True, choices=["shopify", "etsy", "ebay"])
    parser.add_argument("--adapter",     type=str,  required=True, choices=["ip_adapter", "lora"])
    parser.add_argument("--adapter_ckpt",type=Path, required=True)
    parser.add_argument("--prompt",      type=str,  default="a professional product photograph")
    parser.add_argument("--output",      type=Path, default=None)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--steps",       type=int,  default=30)
    parser.add_argument("--scale",       type=float,default=1.0)
    parser.add_argument("--device",      type=str,  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate_product_image(
        product_path=args.product,
        platform=args.platform,
        adapter_type=args.adapter,
        adapter_ckpt=args.adapter_ckpt,
        prompt=args.prompt,
        adapter_scale=args.scale,
        num_inference_steps=args.steps,
        seed=args.seed,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
