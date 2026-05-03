#!/usr/bin/env python3
"""Generate and evaluate images for shortlisted adapter sweep candidates."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)
from omegaconf import OmegaConf
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import open_clip

from adapters.ip_adapter.model import IPAdapterSDXL
from adapters.lora.layers import LoRALinear
from adapters.lora.model import load_lora_weights
from inference.generate import (
    CLIP_IMAGE_ENCODER,
    CLIP_TRANSFORM,
    DEFAULT_PLATFORM_PROMPTS,
    SDXL_BASE,
    VAE_FIX,
    build_reference_product_image,
    mask_to_controlnet_conditioning,
    normalize_foreground_mask,
)
from segmentation import SAM2Extractor, U2NetExtractor


PLATFORMS = ("ebay", "etsy", "shopify")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts"


@dataclass(frozen=True)
class Candidate:
    adapter: str
    platform: str
    candidate: str
    run: str
    checkpoint_path: Path
    val_loss: float | None


@dataclass(frozen=True)
class ProductCase:
    platform: str
    path: Path
    case_id: str


class FullImageMaskExtractor:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def extract(self, image: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        return np.ones(image.shape[:2], dtype=bool)


def dtype_from_name(name: str, device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_shortlist(summary_path: Path) -> list[Candidate]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary["runs"]
    selected: list[Candidate] = []
    for row in rows:
        is_baseline = row.get("candidate") == "baseline"
        is_winner = row.get("rank_within_adapter_platform") == 1
        if not (is_baseline or is_winner):
            continue
        selected.append(
            Candidate(
                adapter=row["adapter"],
                platform=row["platform"],
                candidate=row["candidate"],
                run=row["run"],
                checkpoint_path=Path(row["checkpoint_path"]),
                val_loss=row.get("val_loss"),
            )
        )
    return sorted(selected, key=lambda item: (item.adapter, item.platform, item.candidate != "baseline", item.run))


def resolve_manifest_path(platform_dir: Path, raw_image_path: str) -> Path | None:
    source = Path(raw_image_path)
    candidates = [platform_dir / f"{source.parent.name}_{source.name}", platform_dir / source.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def select_products(platform: str, count: int, seed: int) -> list[ProductCase]:
    platform_dir = Path("data/platform_sets") / platform
    manifest_path = Path("data/platform_sets/manifests") / f"{platform}_val.csv"
    resolved: list[Path] = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = resolve_manifest_path(platform_dir, row["image_path"])
                if path is not None:
                    resolved.append(path)
    if not resolved:
        resolved = sorted(path for path in platform_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    if len(resolved) < count:
        raise ValueError(f"Need {count} products for {platform}, found {len(resolved)}")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(resolved, count), key=lambda path: path.name)
    return [ProductCase(platform=platform, path=path, case_id=f"{platform}_{index:02d}_{path.stem}") for index, path in enumerate(chosen)]


def build_base_components(dtype: torch.dtype, local_files_only: bool) -> dict[str, Any]:
    scheduler = DDPMScheduler.from_pretrained(
        SDXL_BASE,
        subfolder="scheduler",
        local_files_only=local_files_only,
    )
    return {
        "vae": AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=dtype, local_files_only=local_files_only),
        "text_encoder": CLIPTextModel.from_pretrained(
            SDXL_BASE,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ),
        "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(
            SDXL_BASE,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ),
        "tokenizer": CLIPTokenizer.from_pretrained(
            SDXL_BASE,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        ),
        "tokenizer_2": CLIPTokenizer.from_pretrained(
            SDXL_BASE,
            subfolder="tokenizer_2",
            local_files_only=local_files_only,
        ),
        "unet": UNet2DConditionModel.from_pretrained(
            SDXL_BASE,
            subfolder="unet",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ),
        "scheduler": UniPCMultistepScheduler.from_config(scheduler.config),
    }


def build_pipeline_from_components(
    *,
    dtype: torch.dtype,
    controlnet_model_id: str | None,
    strict_controlnet: bool,
    local_files_only: bool,
):
    components = build_base_components(dtype, local_files_only)
    warning = None
    if controlnet_model_id:
        try:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
            return StableDiffusionXLControlNetPipeline(controlnet=controlnet, **components), True, None
        except Exception as exc:
            if strict_controlnet:
                raise
            warning = f"ControlNet fallback: {exc}"
    return StableDiffusionXLPipeline(**components), False, warning


def build_pipeline(
    *,
    device: str,
    dtype: torch.dtype,
    controlnet_model_id: str | None,
    strict_controlnet: bool,
    local_files_only: bool,
):
    vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=dtype, local_files_only=local_files_only)
    controlnet_enabled = False
    warning = None
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
            controlnet_enabled = True
        except Exception as exc:
            if strict_controlnet:
                raise
            warning = f"ControlNet fallback: {exc}"
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_BASE,
                    vae=vae,
                    torch_dtype=dtype,
                    local_files_only=local_files_only,
                )
            except Exception:
                pipe, controlnet_enabled, component_warning = build_pipeline_from_components(
                    dtype=dtype,
                    controlnet_model_id=None,
                    strict_controlnet=strict_controlnet,
                    local_files_only=local_files_only,
                )
                warning = warning or component_warning
    else:
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                SDXL_BASE,
                vae=vae,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
        except Exception:
            pipe, controlnet_enabled, warning = build_pipeline_from_components(
                dtype=dtype,
                controlnet_model_id=None,
                strict_controlnet=strict_controlnet,
                local_files_only=local_files_only,
            )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, controlnet_enabled, warning


def build_mask_extractor(args: argparse.Namespace):
    try:
        extractor = SAM2Extractor(
            checkpoint=args.sam2_checkpoint,
            model_cfg=args.sam2_config,
            device=args.segmentation_device or args.device,
        )
        return extractor, "sam2", True
    except Exception as exc:
        print(f"[warn] sam2_unavailable: {exc}", flush=True)
    try:
        extractor = U2NetExtractor(
            checkpoint=args.u2net_checkpoint,
            device=args.segmentation_device or args.device,
            threshold=args.u2net_threshold,
        )
        return extractor, "u2net", False
    except Exception as exc:
        reason = f"full_image_fallback: {exc}"
        print(f"[warn] {reason}", flush=True)
        return FullImageMaskExtractor(reason), reason, False


def resolve_control_image_mode(args: argparse.Namespace) -> str:
    if args.control_image_mode != "auto":
        return args.control_image_mode
    if "canny" in (args.controlnet_model or "").lower():
        return "canny"
    return "mask"


def image_to_canny_conditioning(
    image: Image.Image,
    mask: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    mask_bool = mask.astype(bool)
    if mask_bool.shape == rgb.shape[:2]:
        rgb = rgb.copy()
        rgb[~mask_bool] = 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def prepare_product(
    product: ProductCase,
    extractor,
    normalize_mask: bool,
    control_image_mode: str,
    canny_low_threshold: int,
    canny_high_threshold: int,
) -> dict[str, Any]:
    image = Image.open(product.path).convert("RGB")
    mask = extractor.extract(np.array(image))
    if normalize_mask:
        mask = normalize_foreground_mask(mask)
    if control_image_mode == "canny":
        control_image = image_to_canny_conditioning(
            image,
            mask,
            canny_low_threshold,
            canny_high_threshold,
        )
    else:
        control_image = mask_to_controlnet_conditioning(mask)
    return {
        "product": product,
        "image": image,
        "mask": mask,
        "control_image": control_image,
        "reference_image": build_reference_product_image(image, mask),
    }


@torch.no_grad()
def ip_hidden_states(adapter: IPAdapterSDXL, reference_image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    clip_input = CLIP_TRANSFORM(reference_image).unsqueeze(0).to(device=device, dtype=dtype)
    cond_ip, uncond_ip = adapter.encode_image(clip_input)
    return torch.cat([uncond_ip, cond_ip], dim=0)


def apply_lora_inference_scale(unet: torch.nn.Module, scale: float) -> None:
    for module in unet.modules():
        if isinstance(module, LoRALinear):
            module.scaling = (module.alpha / module.rank) * scale


def generate_images(args: argparse.Namespace, candidates: list[Candidate], cases: dict[str, list[ProductCase]]) -> list[dict[str, Any]]:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_name(args.dtype, args.device)
    controlnet_model_id = None if args.disable_controlnet else args.controlnet_model
    extractor, segmentation_status, normalize_mask = build_mask_extractor(args)
    control_image_mode = resolve_control_image_mode(args)
    prepared = {
        platform: [
            prepare_product(
                case,
                extractor,
                normalize_mask,
                control_image_mode,
                args.canny_low_threshold,
                args.canny_high_threshold,
            )
            for case in platform_cases
        ]
        for platform, platform_cases in cases.items()
    }

    manifest_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] generating {candidate.adapter}/{candidate.run}", flush=True)
        pipe, controlnet_enabled, warning = build_pipeline(
            device=args.device,
            dtype=dtype,
            controlnet_model_id=controlnet_model_id,
            strict_controlnet=args.strict_controlnet,
            local_files_only=args.local_files_only,
        )
        ip_adapter = None
        if candidate.adapter == "ip_adapter":
            ip_adapter = IPAdapterSDXL.load_pretrained(
                unet=pipe.unet,
                load_directory=candidate.checkpoint_path,
                image_encoder_id=CLIP_IMAGE_ENCODER,
                adapter_scale=args.adapter_scale,
                local_files_only=args.local_files_only,
            ).to(device=args.device, dtype=dtype)
        else:
            load_lora_weights(pipe.unet, candidate.checkpoint_path)
            apply_lora_inference_scale(pipe.unet, args.lora_scale)
            pipe.unet.to(device=args.device, dtype=dtype)

        for prepared_case in prepared[candidate.platform]:
            product = prepared_case["product"]
            for seed in args.seeds:
                out_dir = output_root / candidate.adapter / candidate.run
                out_path = out_dir / f"{product.case_id}_seed{seed}.png"
                if not out_path.exists() or args.overwrite:
                    generator = torch.Generator(device=args.device).manual_seed(seed)
                    prompt = DEFAULT_PLATFORM_PROMPTS[candidate.platform]
                    generate_kwargs: dict[str, Any] = {
                        "prompt": prompt,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "num_inference_steps": args.steps,
                        "guidance_scale": args.guidance_scale,
                        "height": args.height,
                        "width": args.width,
                        "generator": generator,
                    }
                    if controlnet_enabled:
                        generate_kwargs["image"] = prepared_case["control_image"]
                        generate_kwargs["controlnet_conditioning_scale"] = args.controlnet_conditioning_scale
                    if ip_adapter is not None:
                        generate_kwargs["cross_attention_kwargs"] = {
                            "ip_hidden_states": ip_hidden_states(
                                ip_adapter,
                                prepared_case["reference_image"],
                                args.device,
                                dtype,
                            )
                        }
                    out_dir.mkdir(parents=True, exist_ok=True)
                    pipe(**generate_kwargs).images[0].save(out_path)
                manifest_rows.append(
                    {
                        "adapter": candidate.adapter,
                        "platform": candidate.platform,
                        "candidate": candidate.candidate,
                        "run": candidate.run,
                        "checkpoint_path": str(candidate.checkpoint_path),
                        "val_loss": candidate.val_loss,
                        "product_path": str(product.path),
                        "case_id": product.case_id,
                        "seed": seed,
                        "prompt": DEFAULT_PLATFORM_PROMPTS[candidate.platform],
                        "steps": args.steps,
                        "height": args.height,
                        "width": args.width,
                        "guidance_scale": args.guidance_scale,
                        "adapter_scale": args.adapter_scale if candidate.adapter == "ip_adapter" else None,
                        "lora_scale": args.lora_scale if candidate.adapter == "lora" else None,
                        "output_path": str(out_path),
                        "controlnet_enabled": controlnet_enabled,
                        "control_image_mode": control_image_mode if controlnet_enabled else None,
                        "controlnet_warning": warning,
                        "segmentation_status": segmentation_status,
                    }
                )
        del pipe
        del ip_adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest_rows


@torch.no_grad()
def embed_paths(model, preprocess, paths: list[Path], device: str, batch_size: int) -> np.ndarray:
    feats = []
    for start in tqdm(range(0, len(paths), batch_size), desc="CLIP embedding", leave=False):
        batch = []
        for path in paths[start : start + batch_size]:
            batch.append(preprocess(Image.open(path).convert("RGB")))
        inp = torch.stack(batch).to(device)
        out = model.encode_image(inp)
        out = F.normalize(out, dim=-1)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 768))


def sample_reference_paths(platform: str, limit: int, seed: int) -> list[Path]:
    paths = sorted(path for path in (Path("data/platform_sets") / platform).rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    if len(paths) <= limit:
        return paths
    rng = random.Random(seed)
    return sorted(rng.sample(paths, limit), key=lambda path: path.name)


def pairwise_diversity(embs: np.ndarray) -> dict[str, float]:
    if len(embs) < 2:
        return {"clip_diversity_mean_pairwise_distance": 0.0, "clip_diversity_mean_pairwise_similarity": 1.0 if len(embs) else 0.0}
    sims = embs @ embs.T
    upper = np.triu_indices(len(embs), k=1)
    values = sims[upper]
    return {
        "clip_diversity_mean_pairwise_distance": float((1.0 - values).mean()),
        "clip_diversity_mean_pairwise_similarity": float(values.mean()),
    }


def evaluate_manifest(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(args.device).eval()

    ref_embeddings: dict[str, np.ndarray] = {}
    for platform in PLATFORMS:
        paths = sample_reference_paths(platform, args.reference_limit, args.selection_seed)
        ref_embeddings[platform] = embed_paths(model, preprocess, paths, args.device, args.clip_batch_size)
        print(f"[{platform}] reference embeddings: {ref_embeddings[platform].shape}", flush=True)

    centroids = {}
    for platform, embs in ref_embeddings.items():
        centroid = embs.mean(axis=0, keepdims=True)
        centroids[platform] = centroid / (np.linalg.norm(centroid) + 1e-8)

    train_x = np.concatenate([ref_embeddings[platform] for platform in PLATFORMS], axis=0)
    train_y = [platform for platform in PLATFORMS for _ in range(len(ref_embeddings[platform]))]
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(train_x, train_y)

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["adapter"], row["platform"], row["candidate"], row["run"])
        groups.setdefault(key, []).append(row)

    candidate_results = []
    for (adapter, platform, candidate, run), group in sorted(groups.items()):
        image_paths = [Path(row["output_path"]) for row in group]
        product_paths = [Path(row["product_path"]) for row in group]
        gen_embs = embed_paths(model, preprocess, image_paths, args.device, args.clip_batch_size)
        product_embs = embed_paths(model, preprocess, product_paths, args.device, args.clip_batch_size)
        sims_by_platform = {
            ref_platform: float((gen_embs @ centroid.T).mean())
            for ref_platform, centroid in centroids.items()
        }
        other_sims = [score for ref_platform, score in sims_by_platform.items() if ref_platform != platform]
        preds = knn.predict(gen_embs)
        product_sims = (gen_embs * product_embs).sum(axis=1)
        diversity = pairwise_diversity(gen_embs)
        candidate_results.append(
            {
                "adapter": adapter,
                "platform": platform,
                "candidate": candidate,
                "run": run,
                "sample_count": len(group),
                "val_loss": group[0].get("val_loss"),
                "mean_cosine_sim_target": sims_by_platform[platform],
                "target_margin_vs_best_other": sims_by_platform[platform] - max(other_sims),
                "knn_platform_accuracy": float(accuracy_score([platform] * len(preds), preds)),
                "source_product_clip_similarity": float(product_sims.mean()),
                "source_product_clip_similarity_min": float(product_sims.min()),
                "clip_platform_sims": sims_by_platform,
                **diversity,
            }
        )

    return {
        "manifest_path": str(args.manifest_path),
        "aesthetic_scoring": {
            "available": Path("checkpoints/aesthetic_predictor_v2.pth").exists(),
            "status": "not_run_missing_checkpoint" if not Path("checkpoints/aesthetic_predictor_v2.pth").exists() else "not_run_in_shortlist_harness",
        },
        "fid": {
            "status": "not_run_shortlist_sample_too_small",
            "minimum_recommended_images": args.fid_min_images,
        },
        "results": candidate_results,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def metric_score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        row["target_margin_vs_best_other"],
        row["source_product_clip_similarity"],
        row["mean_cosine_sim_target"],
    )


def write_eval_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["results"]
    lines = ["# Adapter Shortlist Image Eval", ""]
    lines.append(f"Manifest: `{payload['manifest_path']}`")
    lines.append(f"Aesthetic scoring: {payload['aesthetic_scoring']['status']}")
    lines.append(f"FID: {payload['fid']['status']}")
    lines.append("")
    lines.append("## Best By Platform")
    lines.append("")
    lines.append("| Platform | Selected | Margin | Product Sim | Target Sim | kNN Acc |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for platform in PLATFORMS:
        platform_rows = [row for row in rows if row["platform"] == platform]
        best = max(platform_rows, key=metric_score)
        lines.append(
            f"| {platform} | {best['adapter']}/{best['candidate']} | {best['target_margin_vs_best_other']:.6f} | "
            f"{best['source_product_clip_similarity']:.6f} | {best['mean_cosine_sim_target']:.6f} | "
            f"{best['knn_platform_accuracy']:.3f} |"
        )
    lines.append("")
    lines.append("## All Candidates")
    lines.append("")
    lines.append("| Platform | Adapter | Candidate | Samples | Val Loss | Target Sim | Margin | Product Sim | Diversity Dist | kNN Acc |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: (item["platform"], item["adapter"], item["candidate"])):
        lines.append(
            f"| {row['platform']} | {row['adapter']} | {row['candidate']} | {row['sample_count']} | "
            f"{row['val_loss']:.6f} | {row['mean_cosine_sim_target']:.6f} | "
            f"{row['target_margin_vs_best_other']:.6f} | {row['source_product_clip_similarity']:.6f} | "
            f"{row['clip_diversity_mean_pairwise_distance']:.6f} | {row['knn_platform_accuracy']:.3f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    base_cfg = OmegaConf.load("configs/base.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/adapter_quality_screen_250_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapter_shortlist_250"))
    parser.add_argument("--manifest-path", type=Path, default=Path("results/adapter_shortlist_250_manifest.json"))
    parser.add_argument("--eval-json", type=Path, default=Path("results/adapter_shortlist_250_eval.json"))
    parser.add_argument("--eval-md", type=Path, default=Path("results/adapter_shortlist_250_eval.md"))
    parser.add_argument("--products-per-platform", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--seeds", type=parse_seeds, default=[42])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.8)
    parser.add_argument("--control-image-mode", choices=("auto", "mask", "canny"), default="auto")
    parser.add_argument("--canny-low-threshold", type=int, default=100)
    parser.add_argument("--canny-high-threshold", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--segmentation-device", type=str, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--disable-controlnet", action="store_true")
    parser.add_argument("--controlnet-model", type=str, default=base_cfg.model.controlnet)
    parser.add_argument("--strict-controlnet", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path(SAM2Extractor.DEFAULT_CHECKPOINT))
    parser.add_argument("--sam2-config", type=str, default=SAM2Extractor.DEFAULT_CONFIG)
    parser.add_argument("--u2net-checkpoint", type=Path, default=Path("checkpoints/u2net.pth"))
    parser.add_argument("--u2net-threshold", type=float, default=0.5)
    parser.add_argument("--reference-limit", type=int, default=256)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--fid-min-images", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--adapter-filter", choices=("ip_adapter", "lora"), default=None)
    parser.add_argument("--platform-filter", choices=PLATFORMS, default=None)
    parser.add_argument("--candidate-filter", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    candidates = load_shortlist(args.summary)
    if args.adapter_filter is not None:
        candidates = [candidate for candidate in candidates if candidate.adapter == args.adapter_filter]
    if args.platform_filter is not None:
        candidates = [candidate for candidate in candidates if candidate.platform == args.platform_filter]
    if args.candidate_filter is not None:
        candidates = [candidate for candidate in candidates if candidate.candidate == args.candidate_filter]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    cases = {platform: select_products(platform, args.products_per_platform, args.selection_seed) for platform in PLATFORMS}
    shortlist_payload = {
        "candidates": [candidate.__dict__ | {"checkpoint_path": str(candidate.checkpoint_path)} for candidate in candidates],
        "products": {
            platform: [{"case_id": case.case_id, "path": str(case.path)} for case in platform_cases]
            for platform, platform_cases in cases.items()
        },
        "seeds": args.seeds,
        "generation": {
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
            "ip_adapter_scale": args.adapter_scale,
            "lora_scale": args.lora_scale,
            "controlnet_enabled_requested": not args.disable_controlnet,
        },
    }
    write_json(Path("results/adapter_shortlist_250_selection.json"), shortlist_payload)

    if args.skip_generation:
        rows = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    else:
        rows = generate_images(args, candidates, cases)
        write_json(args.manifest_path, rows)
        print(f"wrote {args.manifest_path}", flush=True)

    if not args.skip_eval:
        eval_payload = evaluate_manifest(args, rows)
        write_json(args.eval_json, eval_payload)
        write_eval_markdown(args.eval_md, eval_payload)
        print(f"wrote {args.eval_json}", flush=True)
        print(f"wrote {args.eval_md}", flush=True)


if __name__ == "__main__":
    main()