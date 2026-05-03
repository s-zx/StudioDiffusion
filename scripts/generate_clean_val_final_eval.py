#!/usr/bin/env python3
"""Generate final eval images from leakage-free platform validation splits."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


NEGATIVE_PROMPT = (
    "white outline, bright rim, halo, cutout edge, pasted object, extra product, "
    "duplicate product, extra scene object, extra props, extra text, watermark, blurry, "
    "low quality, distorted, artifacts, fake label"
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    platform: str
    adapter: str
    checkpoint: Path
    adapter_scale: float = 0.1


@dataclass(frozen=True)
class ProductSpec:
    platform: str
    case_id: str
    product_path: Path
    category: str
    source_row_index: int
    original_resolved_path: str


@dataclass(frozen=True)
class RunSpec:
    run_index: int
    model_id: str
    platform: str
    adapter: str
    checkpoint: Path
    adapter_scale: float
    case_id: str
    category: str
    product_path: Path
    output_dir: Path
    clean_composite_path: Path
    log_path: Path


MODELS = [
    ModelSpec(
        model_id="etsy_lora_final_s3000",
        platform="etsy",
        adapter="lora",
        checkpoint=Path("checkpoints/lora/etsy_final_lora_baseline_canny_s3000/final"),
    ),
    ModelSpec(
        model_id="etsy_ip_adapter_final_s3000",
        platform="etsy",
        adapter="ip_adapter",
        checkpoint=Path("checkpoints/ip_adapter/etsy/final"),
    ),
    ModelSpec(
        model_id="ebay_lora_lr2e-4_s3000",
        platform="ebay",
        adapter="lora",
        checkpoint=Path("checkpoints/lora/ebay_lr2e-4_s3000/final"),
    ),
    ModelSpec(
        model_id="ebay_ip_adapter_final_s3000",
        platform="ebay",
        adapter="ip_adapter",
        checkpoint=Path("checkpoints/ip_adapter/ebay_final_ip_baseline_canny_s3000/final"),
    ),
    ModelSpec(
        model_id="shopify_lora_final_s3000",
        platform="shopify",
        adapter="lora",
        checkpoint=Path("checkpoints/lora/shopify/final"),
    ),
    ModelSpec(
        model_id="shopify_ip_adapter_final_s3000",
        platform="shopify",
        adapter="ip_adapter",
        checkpoint=Path("checkpoints/ip_adapter/shopify/final"),
    ),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned.strip("_") or "uncategorized"


def load_products(clean_root: Path, platform: str) -> list[ProductSpec]:
    manifest = clean_root / "manifests" / f"{platform}_val_only.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    products: list[ProductSpec] = []
    for index, row in enumerate(rows, 1):
        product_path = REPO_ROOT / row["image_path"]
        if not product_path.exists():
            raise FileNotFoundError(f"Missing clean val image: {product_path}")
        case_id = f"{platform}_{index:03d}_{Path(row['filename']).stem}"
        products.append(
            ProductSpec(
                platform=platform,
                case_id=case_id,
                product_path=product_path,
                category=row.get("category", "").strip() or "uncategorized",
                source_row_index=int(row["source_row_index"]),
                original_resolved_path=row["original_resolved_path"],
            )
        )
    return products


def build_runs(clean_root: Path, output_root: Path) -> list[RunSpec]:
    products_by_platform = {
        platform: load_products(clean_root, platform)
        for platform in ("etsy", "ebay", "shopify")
    }
    runs: list[RunSpec] = []
    for model in MODELS:
        checkpoint = REPO_ROOT / model.checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model.model_id}: {checkpoint}")
        for product in products_by_platform[model.platform]:
            run_index = len(runs) + 1
            output_dir = output_root / "runs" / model.model_id / product.case_id
            image_name = f"{model.model_id}__{product.case_id}__{safe_part(product.category)}.png"
            clean_path = output_root / "clean_composites" / model.platform / model.adapter / image_name
            log_path = output_root / "logs" / f"{run_index:04d}_{model.model_id}_{product.case_id}.log"
            runs.append(
                RunSpec(
                    run_index=run_index,
                    model_id=model.model_id,
                    platform=model.platform,
                    adapter=model.adapter,
                    checkpoint=checkpoint,
                    adapter_scale=model.adapter_scale,
                    case_id=product.case_id,
                    category=product.category,
                    product_path=product.product_path,
                    output_dir=output_dir,
                    clean_composite_path=clean_path,
                    log_path=log_path,
                )
            )
    return runs


def command_for_run(run: RunSpec, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "inference/inpaint_composite.py",
        "--product",
        rel(run.product_path),
        "--platform",
        run.platform,
        "--output-dir",
        rel(run.output_dir),
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--strength",
        str(args.strength),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--segmentation-device",
        args.segmentation_device,
        "--dtype",
        args.dtype,
        "--product-scale",
        str(args.product_scale),
        "--controlnet-model",
        args.controlnet_model,
        "--control-image-mode",
        "canny",
        "--controlnet-scale",
        str(args.controlnet_scale),
        "--control-guidance-start",
        str(args.control_guidance_start),
        "--control-guidance-end",
        str(args.control_guidance_end),
        "--canny-low-threshold",
        str(args.canny_low_threshold),
        "--canny-high-threshold",
        str(args.canny_high_threshold),
        "--inpaint-mask-erode",
        str(args.inpaint_mask_erode),
        "--composite-mask-erode",
        str(args.composite_mask_erode),
        "--feather-radius",
        str(args.feather_radius),
        "--dehalo-edge-radius",
        str(args.dehalo_edge_radius),
        "--dehalo-brightness-threshold",
        str(args.dehalo_brightness_threshold),
        "--local-files-only",
    ]
    if run.adapter == "lora":
        command.extend(["--lora-ckpt", rel(run.checkpoint), "--lora-scale", str(run.adapter_scale)])
    else:
        command.extend([
            "--ip-adapter-ckpt",
            rel(run.checkpoint),
            "--ip-adapter-scale",
            str(run.adapter_scale),
        ])
    return command


def write_plan(output_root: Path, runs: list[RunSpec], args: argparse.Namespace) -> None:
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    model_rows = [
        {
            **asdict(model),
            "checkpoint": rel(REPO_ROOT / model.checkpoint),
        }
        for model in MODELS
    ]
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Leakage-free final eval generation over all clean val-only platform images with both LoRA and IP-Adapter checkpoints.",
        "clean_split_summary": rel(REPO_ROOT / args.clean_root / "summary.json"),
        "output_root": rel(output_root),
        "expected_runs": len(runs),
        "settings": {
            "seed": args.seed,
            "steps": args.steps,
            "strength": args.strength,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "device": args.device,
            "segmentation_device": args.segmentation_device,
            "dtype": args.dtype,
            "product_scale": args.product_scale,
            "controlnet_model": args.controlnet_model,
            "controlnet_scale": args.controlnet_scale,
            "control_guidance_start": args.control_guidance_start,
            "control_guidance_end": args.control_guidance_end,
            "canny_low_threshold": args.canny_low_threshold,
            "canny_high_threshold": args.canny_high_threshold,
            "inpaint_mask_erode": args.inpaint_mask_erode,
            "composite_mask_erode": args.composite_mask_erode,
            "feather_radius": args.feather_radius,
            "dehalo_edge_radius": args.dehalo_edge_radius,
            "dehalo_brightness_threshold": args.dehalo_brightness_threshold,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "models": model_rows,
        "runs": [
            {
                **asdict(run),
                "checkpoint": rel(run.checkpoint),
                "product_path": rel(run.product_path),
                "output_dir": rel(run.output_dir),
                "clean_composite_path": rel(run.clean_composite_path),
                "log_path": rel(run.log_path),
            }
            for run in runs
        ],
    }
    (metadata / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def write_manifest(output_root: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    manifest = metadata / "final_outputs_manifest.csv"
    fieldnames = [
        "run_index",
        "status",
        "platform",
        "adapter",
        "model_id",
        "case_id",
        "category",
        "product_path",
        "checkpoint",
        "clean_composite_path",
        "full_run_dir",
        "log_path",
        "elapsed_seconds",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_status(output_root: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    metadata = output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "total": len(rows),
        "rows": rows,
    }
    (metadata / "generation_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_one(run: RunSpec, args: argparse.Namespace) -> tuple[str, float]:
    import time

    run.clean_composite_path.parent.mkdir(parents=True, exist_ok=True)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    existing_composite = run.output_dir / "composite.png"
    if args.skip_existing and run.clean_composite_path.exists():
        return "skipped", 0.0
    if args.skip_existing and existing_composite.exists():
        shutil.copy2(existing_composite, run.clean_composite_path)
        return "copied_existing", 0.0

    start = time.monotonic()
    command = command_for_run(run, args)
    with run.log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        return f"failed:{result.returncode}", elapsed
    if not existing_composite.exists():
        return "failed:missing_composite", elapsed
    shutil.copy2(existing_composite, run.clean_composite_path)
    return "generated", elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=Path("data/platform_sets_clean"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/final_eval_clean_val_20260429"))
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=8.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--segmentation-device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--product-scale", type=float, default=0.45)
    parser.add_argument("--controlnet-model", type=str, default="diffusers/controlnet-canny-sdxl-1.0")
    parser.add_argument("--controlnet-scale", type=float, default=0.05)
    parser.add_argument("--control-guidance-start", type=float, default=0.0)
    parser.add_argument("--control-guidance-end", type=float, default=0.25)
    parser.add_argument("--canny-low-threshold", type=int, default=120)
    parser.add_argument("--canny-high-threshold", type=int, default=240)
    parser.add_argument("--inpaint-mask-erode", type=int, default=6)
    parser.add_argument("--composite-mask-erode", type=int, default=5)
    parser.add_argument("--feather-radius", type=float, default=1.0)
    parser.add_argument("--dehalo-edge-radius", type=int, default=0)
    parser.add_argument("--dehalo-brightness-threshold", type=int, default=235)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--limit-per-model", type=int, default=None)
    parser.add_argument("--only-model-id", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = REPO_ROOT / args.output_root
    clean_root = REPO_ROOT / args.clean_root
    if not clean_root.exists():
        raise FileNotFoundError(f"Missing clean split root: {clean_root}")
    runs = build_runs(clean_root, output_root)
    if args.only_model_id is not None:
        requested = set(args.only_model_id)
        runs = [run for run in runs if run.model_id in requested]
        missing = requested - {run.model_id for run in runs}
        if missing:
            raise ValueError(f"Unknown --only-model-id value(s): {sorted(missing)}")
    if args.limit_per_model is not None:
        counts: dict[str, int] = {}
        limited_runs = []
        for run in runs:
            count = counts.get(run.model_id, 0)
            if count >= args.limit_per_model:
                continue
            counts[run.model_id] = count + 1
            limited_runs.append(run)
        runs = limited_runs
    output_root.mkdir(parents=True, exist_ok=True)
    write_plan(output_root, runs, args)
    print(f"planned_runs={len(runs)}")
    print(f"output_root={rel(output_root)}")
    if args.dry_run:
        return

    rows: list[dict[str, str | int | float | bool]] = []
    for run in runs:
        print(f"[{run.run_index}/{len(runs)}] {run.model_id} {run.case_id}", flush=True)
        status, elapsed = run_one(run, args)
        row = {
            "run_index": run.run_index,
            "status": status,
            "platform": run.platform,
            "adapter": run.adapter,
            "model_id": run.model_id,
            "case_id": run.case_id,
            "category": run.category,
            "product_path": rel(run.product_path),
            "checkpoint": rel(run.checkpoint),
            "clean_composite_path": rel(run.clean_composite_path),
            "full_run_dir": rel(run.output_dir),
            "log_path": rel(run.log_path),
            "elapsed_seconds": round(elapsed, 3),
        }
        rows.append(row)
        write_manifest(output_root, rows)
        write_status(output_root, rows)
        print(f"  status={status} elapsed_seconds={elapsed:.1f}", flush=True)
    write_manifest(output_root, rows)
    write_status(output_root, rows)


if __name__ == "__main__":
    main()