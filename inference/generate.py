"""
End-to-end product image generation.

Pipeline:
  1. Extract foreground mask with SAM2.
  2. Load SDXL with an optional ControlNet branch.
  3. Attach a trained IP-Adapter or LoRA checkpoint.
  4. Generate a platform-targeted product image.

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
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
)
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.ip_adapter.model import IPAdapterSDXL
from adapters.lora.layers import LoRALinear
from adapters.lora.model import load_lora_weights as load_custom_lora_weights
from segmentation import SAM2Extractor

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

DEFAULT_PLATFORM_PROMPTS = {
    "shopify": "a professional product photograph on a clean white background",
    "etsy": "a warm handcrafted lifestyle product photograph with natural styling",
    "ebay": "a clear detailed marketplace product photograph with neutral background",
}


def mask_to_controlnet_conditioning(mask: np.ndarray) -> Image.Image:
    """Convert binary mask to a 3-channel conditioning image for ControlNet."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    rgb = np.stack([mask_uint8] * 3, axis=-1)
    return Image.fromarray(rgb)


def image_to_canny_conditioning(
    image: Image.Image,
    mask: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Build Canny conditioning from the masked product on a white background."""
    rgb = np.array(image.convert("RGB"))
    mask_bool = mask.astype(bool)
    if mask_bool.shape == rgb.shape[:2]:
        rgb = rgb.copy()
        rgb[~mask_bool] = 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def resolve_control_image_mode(control_image_mode: str, controlnet_model_id: str | None) -> str:
    """Choose a control image type that matches the configured ControlNet."""
    if control_image_mode != "auto":
        return control_image_mode
    if controlnet_model_id and "canny" in controlnet_model_id.lower():
        return "canny"
    return "mask"


def normalize_foreground_mask(mask: np.ndarray) -> np.ndarray:
    """Heuristic fix for SAM2 auto-masks that sometimes select background."""
    binary = mask.astype(bool)
    border_pixels = np.concatenate(
        [binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]]
    )
    border_fill = border_pixels.mean() if len(border_pixels) else 0.0
    if binary.mean() > 0.5 or border_fill > 0.45:
        binary = ~binary
    return binary


def build_reference_product_image(product_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply the foreground mask, crop to the object, and place it on white."""
    rgb = np.array(product_image.convert("RGB"))
    mask_bool = mask.astype(bool)
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return product_image

    pad_x = max(8, int(0.08 * (xs.max() - xs.min() + 1)))
    pad_y = max(8, int(0.08 * (ys.max() - ys.min() + 1)))
    x0 = max(0, xs.min() - pad_x)
    x1 = min(rgb.shape[1], xs.max() + pad_x + 1)
    y0 = max(0, ys.min() - pad_y)
    y1 = min(rgb.shape[0], ys.max() + pad_y + 1)

    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask_bool[y0:y1, x0:x1]
    crop[~crop_mask] = 255
    return Image.fromarray(crop)


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str, dtype: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device in {"cuda", "mps"} else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def pick_segmentation_device(segmentation_device: str | None, generation_device: str) -> str:
    if segmentation_device:
        return segmentation_device
    if generation_device == "cuda":
        return "cuda"
    return "cpu"


def build_pipeline(
    device: str,
    dtype: torch.dtype,
    controlnet_model_id: str | None,
    strict_controlnet: bool = False,
    local_files_only: bool = False,
):
    try:
        vae = AutoencoderKL.from_pretrained(
            VAE_FIX,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the SDXL VAE. Run `bash scripts/prepare_generation_assets.sh` "
            "or retry with network access enabled."
        ) from exc
    if controlnet_model_id:
        try:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                SDXL_BASE,
                controlnet=controlnet,
                vae=vae,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if strict_controlnet:
                raise
            print(
                f"[warn] Failed to load ControlNet '{controlnet_model_id}': {exc}\n"
                "[warn] Falling back to SDXL without ControlNet."
            )
            controlnet_model_id = None
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_BASE,
                    vae=vae,
                    torch_dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception as base_exc:
                raise RuntimeError(
                    "Failed to load base SDXL after ControlNet fallback. Run "
                    "`bash scripts/prepare_generation_assets.sh` to cache the generation assets."
                ) from base_exc
    else:
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                SDXL_BASE,
                vae=vae,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load base SDXL. Run `bash scripts/prepare_generation_assets.sh` "
                "to cache the generation assets."
            ) from exc

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    return pipe, controlnet_model_id is not None


def build_ip_adapter_hidden_states(
    pipe,
    adapter_ckpt: str | Path,
    reference_image: Image.Image,
    device: str,
    dtype: torch.dtype,
    adapter_scale: float,
    local_files_only: bool = False,
) -> torch.Tensor:
    try:
        adapter = IPAdapterSDXL.load_pretrained(
            unet=pipe.unet,
            load_directory=str(adapter_ckpt),
            image_encoder_id=CLIP_IMAGE_ENCODER,
            adapter_scale=adapter_scale,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the IP-Adapter checkpoint or CLIP vision encoder. "
            "Confirm the checkpoint directory exists and run "
            "`bash scripts/prepare_generation_assets.sh shopify` if assets are missing."
        ) from exc
    adapter = adapter.to(device=device, dtype=dtype)

    clip_input = CLIP_TRANSFORM(reference_image).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        cond_ip, uncond_ip = adapter.encode_image(clip_input)
    return torch.cat([uncond_ip, cond_ip], dim=0)


def apply_lora_inference_scale(unet: torch.nn.Module, scale: float) -> None:
    """Scale custom LoRA branches at inference without changing saved weights."""
    for module in unet.modules():
        if isinstance(module, LoRALinear):
            module.scaling = (module.alpha / module.rank) * scale


def generate_product_image(
    product_path: str | Path,
    platform: str,
    adapter_type: str,
    adapter_ckpt: str | Path,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, artifacts",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    adapter_scale: float = 0.1,
    controlnet_conditioning_scale: float = 0.8,
    control_image_mode: str = "auto",
    canny_low_threshold: int = 100,
    canny_high_threshold: int = 200,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    device: str | None = None,
    segmentation_device: str | None = None,
    dtype: str = "auto",
    controlnet_model_id: str | None = None,
    sam2_checkpoint: str | Path = SAM2Extractor.DEFAULT_CHECKPOINT,
    sam2_model_cfg: str = SAM2Extractor.DEFAULT_CONFIG,
    strict_controlnet: bool = False,
    local_files_only: bool = False,
    mask_output_path: str | Path | None = None,
    control_output_path: str | Path | None = None,
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
    device = pick_device(device)
    segmentation_device = pick_segmentation_device(segmentation_device, device)
    torch_dtype = pick_dtype(device, dtype)
    if not prompt:
        prompt = DEFAULT_PLATFORM_PROMPTS[platform]

    generator = torch.Generator(device=device).manual_seed(seed)
    product_image_pil = Image.open(product_path).convert("RGB")

    # ---- Step 1: Segment product ----
    extractor = SAM2Extractor(
        checkpoint=sam2_checkpoint,
        model_cfg=sam2_model_cfg,
        device=segmentation_device,
    )
    product_image = np.array(product_image_pil)
    mask = normalize_foreground_mask(extractor.extract(product_image))
    resolved_control_image_mode = resolve_control_image_mode(control_image_mode, controlnet_model_id)
    if resolved_control_image_mode == "canny":
        control_image = image_to_canny_conditioning(
            product_image_pil,
            mask,
            canny_low_threshold,
            canny_high_threshold,
        )
    else:
        control_image = mask_to_controlnet_conditioning(mask)
    reference_image = build_reference_product_image(product_image_pil, mask)

    if mask_output_path is not None:
        mask_path = Path(mask_output_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        print(f"Saved mask to {mask_path}")

    if control_output_path is not None:
        control_path = Path(control_output_path)
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_image.save(control_path)
        print(f"Saved control image to {control_path}")

    # ---- Step 2: Load pipeline ----
    pipe, controlnet_enabled = build_pipeline(
        device=device,
        dtype=torch_dtype,
        controlnet_model_id=controlnet_model_id,
        strict_controlnet=strict_controlnet,
        local_files_only=local_files_only,
    )

    # ---- Step 3: Load adapter ----
    cross_attention_kwargs = None
    if adapter_type == "ip_adapter":
        ip_hidden_states = build_ip_adapter_hidden_states(
            pipe=pipe,
            adapter_ckpt=adapter_ckpt,
            reference_image=reference_image,
            device=device,
            dtype=torch_dtype,
            adapter_scale=adapter_scale,
            local_files_only=local_files_only,
        )
        cross_attention_kwargs = {"ip_hidden_states": ip_hidden_states}
    elif adapter_type == "lora":
        load_custom_lora_weights(pipe.unet, adapter_ckpt)
        apply_lora_inference_scale(pipe.unet, adapter_scale)
        pipe.unet.to(device=device, dtype=torch_dtype)
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
        height=height,
        width=width,
        generator=generator,
    )
    if not controlnet_enabled:
        generate_kwargs.pop("image")
        generate_kwargs.pop("controlnet_conditioning_scale")
    if cross_attention_kwargs is not None:
        generate_kwargs["cross_attention_kwargs"] = cross_attention_kwargs

    result = pipe(**generate_kwargs).images[0]

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_path))
        print(f"Saved to {output_path}")

    return result


def main() -> None:
    base_cfg = OmegaConf.load("configs/base.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--product",     type=Path, required=True)
    parser.add_argument("--platform",    type=str,  required=True, choices=["shopify", "etsy", "ebay"])
    parser.add_argument("--adapter",     type=str,  required=True, choices=["ip_adapter", "lora"])
    parser.add_argument("--adapter_ckpt",type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Generation prompt. Defaults to a platform-specific prompt when omitted.",
    )
    parser.add_argument("--output",      type=Path, default=None)
    parser.add_argument("--mask-output", type=Path, default=None)
    parser.add_argument("--control-output", type=Path, default=None)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--steps",       type=int,  default=30)
    parser.add_argument("--scale",       type=float,default=0.1)
    parser.add_argument("--height",      type=int,  default=1024)
    parser.add_argument("--width",       type=int,  default=1024)
    parser.add_argument("--device",      type=str,  default=None)
    parser.add_argument("--segmentation-device", type=str, default=None)
    parser.add_argument("--dtype",       type=str,  default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--disable-controlnet", action="store_true")
    parser.add_argument("--control-image-mode", type=str, default="auto", choices=["auto", "canny", "mask"])
    parser.add_argument("--canny-low-threshold", type=int, default=100)
    parser.add_argument("--canny-high-threshold", type=int, default=200)
    parser.add_argument(
        "--controlnet-model",
        type=str,
        default=base_cfg.model.controlnet,
        help="ControlNet model id. Ignored when --disable-controlnet is set.",
    )
    parser.add_argument(
        "--strict-controlnet",
        action="store_true",
        help="Fail instead of falling back to base SDXL when ControlNet cannot be loaded.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face models from the local cache only.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path(SAM2Extractor.DEFAULT_CHECKPOINT),
        help="Path to the SAM2 checkpoint used for foreground extraction.",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default=SAM2Extractor.DEFAULT_CONFIG,
        help="SAM2 model config name.",
    )
    args = parser.parse_args()

    generate_product_image(
        product_path=args.product,
        platform=args.platform,
        adapter_type=args.adapter,
        adapter_ckpt=args.adapter_ckpt,
        prompt=args.prompt,
        adapter_scale=args.scale,
        control_image_mode=args.control_image_mode,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        device=args.device,
        segmentation_device=args.segmentation_device,
        dtype=args.dtype,
        controlnet_model_id=None if args.disable_controlnet else args.controlnet_model,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_config,
        strict_controlnet=args.strict_controlnet,
        local_files_only=args.local_files_only,
        mask_output_path=args.mask_output,
        control_output_path=args.control_output,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
