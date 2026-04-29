#!/usr/bin/env python3
"""Compute final-eval quality metrics from generated outputs and original inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from evaluation import CLIPDiversity, CLIPPlatformAlignment, FIDScorer

GENERATED_ROOT = ROOT / "final eval clean val"
ORIGINAL_ROOT = ROOT / "final eval original inputs"
RESULTS_DIR = ROOT / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PLATFORMS = ("shopify", "etsy", "ebay")
ADAPTERS = ("ip_adapter", "lora")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=default_device(), help="cpu, mps, or cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "final_eval_metrics.json",
        help="Where to save nested JSON metrics.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RESULTS_DIR / "final_eval_metrics.csv",
        help="Where to save flat CSV metrics.",
    )
    args = parser.parse_args()

    ref_dirs = {
        platform: ORIGINAL_ROOT / "original_eval_inputs" / platform for platform in PLATFORMS
    }

    clip_alignment = CLIPPlatformAlignment(device=args.device)
    clip_alignment.build_reference_embeddings(ref_dirs)
    clip_diversity = CLIPDiversity(device=args.device)
    fid_scorer = FIDScorer(device=args.device)

    metrics_json: dict[str, dict[str, dict[str, float | int | str]]] = {}
    metrics_rows: list[dict[str, object]] = []

    for platform in PLATFORMS:
        metrics_json[platform] = {}
        ref_images = collect_images(ref_dirs[platform])
        for adapter in ADAPTERS:
            gen_dir = GENERATED_ROOT / "clean_composites" / platform / adapter
            gen_images = collect_images(gen_dir)
            clip_metrics = clip_alignment.evaluate(gen_images, target_platform=platform)
            diversity_metrics = clip_diversity.score(gen_images)
            fid_value = fid_scorer.score(gen_images, ref_images)

            combo_metrics: dict[str, float | int | str] = {
                "platform": platform,
                "adapter": adapter,
                "generated_count": len(gen_images),
                "reference_count": len(ref_images),
                **clip_metrics,
                **diversity_metrics,
                "fid": fid_value,
            }
            metrics_json[platform][adapter] = combo_metrics
            metrics_rows.append(combo_metrics)
            print(
                f"[{platform}/{adapter}] "
                f"knn={combo_metrics['knn_accuracy']:.3f} "
                f"clip={combo_metrics['mean_cosine_sim']:.3f} "
                f"fid={combo_metrics['fid']:.3f}"
            )

    args.output_json.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    write_csv(
        args.output_csv,
        metrics_rows,
        [
            "platform",
            "adapter",
            "generated_count",
            "reference_count",
            "mean_cosine_sim",
            "mean_cosine_sim_shopify",
            "mean_cosine_sim_etsy",
            "mean_cosine_sim_ebay",
            "knn_accuracy",
            "clip_diversity_mean_pairwise_distance",
            "clip_diversity_mean_pairwise_similarity",
            "clip_diversity_mean_nearest_neighbor_similarity",
            "clip_diversity_sample_count",
            "fid",
        ],
    )
    print(f"Saved final-eval metrics to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
