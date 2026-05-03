"""Product-preserving SDXL inpainting baseline.

This is the Test 1 path from results/inpaint_composite_test_plan.md:
SAM2 foreground mask -> invert for background inpainting -> composite original
foreground pixels back onto the inpainted background.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.lora.layers import LoRALinear
from adapters.lora.model import load_lora_weights
from inference.generate import (
    build_ip_adapter_hidden_states,
    build_reference_product_image,
    image_to_canny_conditioning,
    mask_to_controlnet_conditioning,
    normalize_foreground_mask,
)
from segmentation import SAM2Extractor

SDXL_INPAINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_CONTROLNET = "diffusers/controlnet-canny-sdxl-1.0"

DEFAULT_PLATFORM_PROMPTS = {
    "etsy": (
        "warm Etsy-style handmade product photo, natural wood tabletop, soft window light, "
        "cozy craft studio background, neutral props, realistic shadows, high quality product photography"
    ),
    "shopify": (
        "clean e-commerce catalog product photo, pure white studio background, soft contact shadow, "
        "centered product, high-key lighting, sharp commercial photography"
    ),
    "amazon": (
        "clean Amazon-style marketplace product photo, pure white studio background, soft contact shadow, "
        "centered product, high-key lighting, sharp commercial photography"
    ),
    "ebay": (
        "clear detailed marketplace product photo, neutral studio background, soft lighting, realistic shadow, "
        "sharp product listing photography"
    ),
}

NEGATIVE_PROMPT = (
    "white outline, bright rim, halo, cutout edge, pasted object, blurry, low quality, "
    "distorted, artifacts, fake label, extra text, watermark"
)


def pick_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, dtype: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def resize_with_padding(
    image: Image.Image,
    size: tuple[int, int],
    fill: int | tuple[int, int, int],
    content_scale: float,
) -> Image.Image:
    target_size = (
        max(1, round(size[0] * content_scale)),
        max(1, round(size[1] * content_scale)),
    )
    contained = ImageOps.contain(image, target_size, Image.Resampling.LANCZOS)
    canvas = Image.new(image.mode, size, fill)
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas


def resize_mask_with_padding(
    mask: np.ndarray,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    content_scale: float,
) -> Image.Image:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    target_size = (
        max(1, round(output_size[0] * content_scale)),
        max(1, round(output_size[1] * content_scale)),
    )
    scale = min(target_size[0] / source_size[0], target_size[1] / source_size[1])
    resized_size = (max(1, round(source_size[0] * scale)), max(1, round(source_size[1] * scale)))
    resized_mask = mask_image.resize(resized_size, Image.Resampling.NEAREST)
    canvas = Image.new("L", output_size, 0)
    offset = ((output_size[0] - resized_size[0]) // 2, (output_size[1] - resized_size[1]) // 2)
    canvas.paste(resized_mask, offset)
    return canvas


def erode_mask(mask_image: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask_image
    return mask_image.filter(ImageFilter.MinFilter(radius * 2 + 1))


def near_white_edge_mask(
    source_image: Image.Image,
    mask_image: Image.Image,
    radius: int,
    brightness_threshold: int,
) -> np.ndarray:
    if radius <= 0:
        return np.zeros(mask_image.size[::-1], dtype=bool)

    mask_l = mask_image.convert("L")
    edge_band = (np.array(mask_l) > 0) & ~(np.array(erode_mask(mask_l, radius)) > 0)
    source = np.array(source_image.convert("RGB"))
    channel_min = source.min(axis=-1)
    channel_max = source.max(axis=-1)
    near_white = (channel_min >= brightness_threshold) & ((channel_max - channel_min) <= 35)
    return edge_band & near_white


def dehalo_source_image(
    source_image: Image.Image,
    fill_image: Image.Image,
    mask_image: Image.Image,
    radius: int,
    brightness_threshold: int,
) -> Image.Image:
    """Replace near-white source-background fringe pixels inside the mask edge.

    SAM2 masks can include a few pixels of the original white product-photo
    background. If those pixels are composited back, they create a bright rim.
    This keeps real product pixels, but swaps near-white edge-band pixels with
    the generated inpainted background.
    """
    if radius <= 0:
        return source_image

    halo = near_white_edge_mask(source_image, mask_image, radius, brightness_threshold)
    source = np.array(source_image.convert("RGB"))
    fill = np.array(fill_image.convert("RGB"))

    result = source.copy()
    result[halo] = fill[halo]
    return Image.fromarray(result)


def mask_to_edge_conditioning(mask_image: Image.Image, dilation: int) -> Image.Image:
    edge = mask_image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge = edge.point(lambda value: 255 if value > 0 else 0)
    if dilation > 0:
        edge = edge.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    return Image.merge("RGB", (edge, edge, edge))


def build_contact_sheet(items: list[tuple[str, Image.Image]], tile_size: tuple[int, int] = (256, 256)) -> Image.Image:
    label_height = 28
    sheet = Image.new("RGB", (tile_size[0] * len(items), tile_size[1] + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        tile = ImageOps.contain(image.convert("RGB"), tile_size, Image.Resampling.LANCZOS)
        x0 = index * tile_size[0] + (tile_size[0] - tile.width) // 2
        y0 = label_height + (tile_size[1] - tile.height) // 2
        sheet.paste(tile, (x0, y0))
        draw.text((index * tile_size[0] + 8, 8), label, fill=(0, 0, 0))
    return sheet


def load_inpaint_pipeline(
    device: str,
    dtype: torch.dtype,
    local_files_only: bool,
    controlnet_model_id: str | None,
) -> StableDiffusionXLInpaintPipeline | StableDiffusionXLControlNetInpaintPipeline:
    vae = AutoencoderKL.from_pretrained(
        VAE_FIX,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    if controlnet_model_id:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            SDXL_INPAINT,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    else:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            SDXL_INPAINT,
            vae=vae,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


def apply_lora_inference_scale(unet: torch.nn.Module, scale: float) -> None:
    for module in unet.modules():
        if isinstance(module, LoRALinear):
            module.scaling = (module.alpha / module.rank) * scale


def run_inpaint_composite(
    product_path: Path,
    platform: str,
    output_dir: Path,
    prompt: str | None,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    strength: float,
    height: int,
    width: int,
    seed: int,
    device: str | None,
    segmentation_device: str | None,
    dtype: str,
    sam2_checkpoint: Path,
    sam2_config: str,
    feather_radius: float,
    product_scale: float,
    inpaint_mask_erode: int,
    composite_mask_erode: int,
    controlnet_model_id: str | None,
    control_image_mode: str,
    controlnet_conditioning_scale: float,
    control_guidance_start: float,
    control_guidance_end: float,
    canny_low_threshold: int,
    canny_high_threshold: int,
    control_edge_dilate: int,
    lora_checkpoint: Path | None,
    lora_scale: float,
    ip_adapter_checkpoint: Path | None,
    ip_adapter_scale: float,
    dehalo_edge_radius: int,
    dehalo_brightness_threshold: int,
    local_files_only: bool,
) -> dict[str, str | int | float]:
    device = pick_device(device)
    segmentation_device = segmentation_device or device
    torch_dtype = pick_dtype(device, dtype)
    prompt = prompt or DEFAULT_PLATFORM_PROMPTS[platform]
    if lora_checkpoint is not None and ip_adapter_checkpoint is not None:
        raise ValueError("Choose either --lora-ckpt or --ip-adapter-ckpt, not both.")

    output_dir.mkdir(parents=True, exist_ok=True)
    original = Image.open(product_path).convert("RGB")

    extractor = SAM2Extractor(checkpoint=sam2_checkpoint, model_cfg=sam2_config, device=segmentation_device)
    foreground_mask = normalize_foreground_mask(extractor.extract(np.array(original)))
    foreground_ratio = float(foreground_mask.mean())

    output_size = (width, height)
    canvas = resize_with_padding(original, output_size, (255, 255, 255), product_scale)
    foreground_mask_image = resize_mask_with_padding(foreground_mask, original.size, output_size, product_scale)
    inpaint_protection_mask = erode_mask(foreground_mask_image, inpaint_mask_erode)
    dehalo_mask = near_white_edge_mask(
        canvas,
        foreground_mask_image,
        dehalo_edge_radius,
        dehalo_brightness_threshold,
    )
    if dehalo_mask.any():
        protection = np.array(inpaint_protection_mask)
        protection[dehalo_mask] = 0
        inpaint_protection_mask = Image.fromarray(protection, mode="L")
    inpaint_mask_image = ImageOps.invert(inpaint_protection_mask)
    resized_foreground_mask = np.array(foreground_mask_image) > 0
    control_image = None
    if controlnet_model_id:
        if control_image_mode == "canny":
            control_image = image_to_canny_conditioning(
                canvas,
                resized_foreground_mask,
                canny_low_threshold,
                canny_high_threshold,
            )
        elif control_image_mode == "mask-edge":
            control_image = mask_to_edge_conditioning(foreground_mask_image, control_edge_dilate)
        else:
            control_image = mask_to_controlnet_conditioning(resized_foreground_mask)

    pipe = load_inpaint_pipeline(
        device=device,
        dtype=torch_dtype,
        local_files_only=local_files_only,
        controlnet_model_id=controlnet_model_id,
    )
    cross_attention_kwargs = None
    ip_adapter_reference = None
    if ip_adapter_checkpoint is not None:
        ip_adapter_reference = build_reference_product_image(original, foreground_mask)
        ip_hidden_states = build_ip_adapter_hidden_states(
            pipe=pipe,
            adapter_ckpt=ip_adapter_checkpoint,
            reference_image=ip_adapter_reference,
            device=device,
            dtype=torch_dtype,
            adapter_scale=ip_adapter_scale,
            local_files_only=local_files_only,
        )
        cross_attention_kwargs = {"ip_hidden_states": ip_hidden_states}
    elif lora_checkpoint is not None:
        load_lora_weights(pipe.unet, lora_checkpoint)
        apply_lora_inference_scale(pipe.unet, lora_scale)
        pipe.unet.to(device=device, dtype=torch_dtype)
    generator = torch.Generator(device=device).manual_seed(seed)

    generate_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": canvas,
        "mask_image": inpaint_mask_image,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "height": height,
        "width": width,
        "generator": generator,
    }
    if control_image is not None:
        generate_kwargs.update(
            {
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            }
        )
    if cross_attention_kwargs is not None:
        generate_kwargs["cross_attention_kwargs"] = cross_attention_kwargs

    with torch.inference_mode():
        inpainted = pipe(**generate_kwargs).images[0]

    source_for_composite = dehalo_source_image(
        canvas,
        inpainted,
        foreground_mask_image,
        dehalo_edge_radius,
        dehalo_brightness_threshold,
    )
    composite_mask = erode_mask(foreground_mask_image, composite_mask_erode)
    if feather_radius > 0:
        composite_mask = composite_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    composite = Image.composite(source_for_composite, inpainted, composite_mask)

    paths = {
        "original_canvas": output_dir / "original_canvas.png",
        "dehalo_source": output_dir / "dehalo_source.png",
        "dehalo_mask": output_dir / "dehalo_edge_mask.png",
        "foreground_mask": output_dir / "sam2_foreground_mask.png",
        "inpaint_protection_mask": output_dir / "inpaint_protection_mask.png",
        "inpaint_mask": output_dir / "inpaint_background_mask.png",
        "composite_mask": output_dir / "composite_alpha_mask.png",
        "control_image": output_dir / "control_image.png",
        "ip_adapter_reference": output_dir / "ip_adapter_reference.png",
        "raw_inpaint": output_dir / "raw_inpaint.png",
        "composite": output_dir / "composite.png",
        "contact_sheet": output_dir / "contact_sheet.png",
        "manifest": output_dir / "manifest.json",
    }
    canvas.save(paths["original_canvas"])
    source_for_composite.save(paths["dehalo_source"])
    Image.fromarray(dehalo_mask.astype(np.uint8) * 255, mode="L").save(paths["dehalo_mask"])
    foreground_mask_image.save(paths["foreground_mask"])
    inpaint_protection_mask.save(paths["inpaint_protection_mask"])
    inpaint_mask_image.save(paths["inpaint_mask"])
    composite_mask.save(paths["composite_mask"])
    if control_image is not None:
        control_image.save(paths["control_image"])
    if ip_adapter_reference is not None:
        ip_adapter_reference.save(paths["ip_adapter_reference"])
    inpainted.save(paths["raw_inpaint"])
    composite.save(paths["composite"])
    contact_items = [
        ("original", canvas),
        ("dehalo src", source_for_composite),
        ("dehalo mask", Image.fromarray(dehalo_mask.astype(np.uint8) * 255, mode="L")),
        ("sam2 mask", foreground_mask_image),
        ("inpaint mask", inpaint_mask_image),
        ("alpha mask", composite_mask),
    ]
    if control_image is not None:
        contact_items.append(("control", control_image))
    if ip_adapter_reference is not None:
        contact_items.append(("ip ref", ip_adapter_reference))
    contact_items.extend([("raw inpaint", inpainted), ("composite", composite)])
    build_contact_sheet(contact_items).save(paths["contact_sheet"])

    manifest = {
        "product_path": str(product_path),
        "platform": platform,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "height": height,
        "width": width,
        "seed": seed,
        "device": device,
        "segmentation_device": segmentation_device,
        "dtype": dtype,
        "sam2_checkpoint": str(sam2_checkpoint),
        "sam2_config": sam2_config,
        "foreground_mask_ratio": foreground_ratio,
        "product_scale": product_scale,
        "inpaint_mask_erode": inpaint_mask_erode,
        "composite_mask_erode": composite_mask_erode,
        "feather_radius": feather_radius,
        "controlnet_model_id": controlnet_model_id,
        "control_image_mode": control_image_mode if controlnet_model_id else None,
        "controlnet_conditioning_scale": controlnet_conditioning_scale if controlnet_model_id else None,
        "control_guidance_start": control_guidance_start if controlnet_model_id else None,
        "control_guidance_end": control_guidance_end if controlnet_model_id else None,
        "canny_low_threshold": canny_low_threshold if controlnet_model_id else None,
        "canny_high_threshold": canny_high_threshold if controlnet_model_id else None,
        "control_edge_dilate": control_edge_dilate if controlnet_model_id else None,
        "lora_checkpoint": str(lora_checkpoint) if lora_checkpoint else None,
        "lora_scale": lora_scale if lora_checkpoint else None,
        "ip_adapter_checkpoint": str(ip_adapter_checkpoint) if ip_adapter_checkpoint else None,
        "ip_adapter_scale": ip_adapter_scale if ip_adapter_checkpoint else None,
        "dehalo_edge_radius": dehalo_edge_radius,
        "dehalo_brightness_threshold": dehalo_brightness_threshold,
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key != "manifest"
            and (key != "control_image" or control_image is not None)
            and (key != "ip_adapter_reference" or ip_adapter_reference is not None)
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 + SDXL background-only inpaint/composite baseline.")
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--platform", type=str, choices=sorted(DEFAULT_PLATFORM_PROMPTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--segmentation-device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path(SAM2Extractor.DEFAULT_CHECKPOINT))
    parser.add_argument("--sam2-config", type=str, default=SAM2Extractor.DEFAULT_CONFIG)
    parser.add_argument("--feather-radius", type=float, default=1.0)
    parser.add_argument("--product-scale", type=float, default=1.0)
    parser.add_argument(
        "--inpaint-mask-erode",
        type=int,
        default=0,
        help="Pixels to erode the protected foreground before inversion, creating an editable seam band. Use 0 for strict mask preservation.",
    )
    parser.add_argument(
        "--composite-mask-erode",
        type=int,
        default=1,
        help="Pixels to erode the foreground alpha before feathering, reducing source-background halos.",
    )
    parser.add_argument(
        "--controlnet-model",
        type=str,
        default=None,
        help=f"Optional ControlNet model id. Use '{DEFAULT_CONTROLNET}' for Test 3 Canny edge control.",
    )
    parser.add_argument("--control-image-mode", type=str, default="canny", choices=["canny", "mask-edge", "mask"])
    parser.add_argument("--controlnet-scale", type=float, default=0.35)
    parser.add_argument("--control-guidance-start", type=float, default=0.0)
    parser.add_argument("--control-guidance-end", type=float, default=0.6)
    parser.add_argument("--canny-low-threshold", type=int, default=100)
    parser.add_argument("--canny-high-threshold", type=int, default=200)
    parser.add_argument("--control-edge-dilate", type=int, default=1)
    parser.add_argument("--lora-ckpt", type=Path, default=None)
    parser.add_argument("--lora-scale", type=float, default=0.25)
    parser.add_argument("--ip-adapter-ckpt", type=Path, default=None)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.25)
    parser.add_argument(
        "--dehalo-edge-radius",
        type=int,
        default=8,
        help="Mask-edge radius to scan for near-white source background fringe before compositing. Use 0 to disable.",
    )
    parser.add_argument(
        "--dehalo-brightness-threshold",
        type=int,
        default=235,
        help="Minimum RGB channel value treated as near-white fringe by --dehalo-edge-radius.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    run_inpaint_composite(
        product_path=args.product,
        platform=args.platform,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        height=args.height,
        width=args.width,
        seed=args.seed,
        device=args.device,
        segmentation_device=args.segmentation_device,
        dtype=args.dtype,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        feather_radius=args.feather_radius,
        product_scale=args.product_scale,
        inpaint_mask_erode=args.inpaint_mask_erode,
        composite_mask_erode=args.composite_mask_erode,
        controlnet_model_id=args.controlnet_model,
        control_image_mode=args.control_image_mode,
        controlnet_conditioning_scale=args.controlnet_scale,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end,
        canny_low_threshold=args.canny_low_threshold,
        canny_high_threshold=args.canny_high_threshold,
        control_edge_dilate=args.control_edge_dilate,
        lora_checkpoint=args.lora_ckpt,
        lora_scale=args.lora_scale,
        ip_adapter_checkpoint=args.ip_adapter_ckpt,
        ip_adapter_scale=args.ip_adapter_scale,
        dehalo_edge_radius=args.dehalo_edge_radius,
        dehalo_brightness_threshold=args.dehalo_brightness_threshold,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()