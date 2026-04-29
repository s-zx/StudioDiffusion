#!/usr/bin/env python3
"""Build SD-30 tables and a report-ready summary from clean eval metadata."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "final eval clean val"
META_DIR = EVAL_DIR / "metadata"
RESULTS_DIR = ROOT / "results"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main() -> None:
    clean_split = load_json(META_DIR / "clean_split_summary.json")
    generation_status = load_json(META_DIR / "generation_status.json")
    run_plan = load_json(META_DIR / "run_plan.json")
    manifest = load_csv(META_DIR / "final_outputs_manifest.csv")

    overfit_paths = {
        "shopify": RESULTS_DIR / "ip_adapter_shopify_overfit.json",
        "etsy": RESULTS_DIR / "ip_adapter_etsy_overfit.json",
        "ebay": RESULTS_DIR / "ip_adapter_ebay_overfit.json",
    }
    overfit = {platform: load_json(path) for platform, path in overfit_paths.items()}

    # Table 1: clean split sizes
    split_rows = []
    for platform in ("shopify", "etsy", "ebay"):
        item = clean_split[platform]
        split_rows.append(
            {
                "platform": platform,
                "clean_train_files": item["clean_train_files"],
                "clean_val_only_files": item["clean_val_only_files"],
                "val_overlap_excluded_files": item["val_overlap_excluded_files"],
                "original_train_rows": item["original_train_rows"],
                "original_val_rows": item["original_val_rows"],
            }
        )
    write_csv(
        RESULTS_DIR / "sd30_clean_split_table.csv",
        split_rows,
        [
            "platform",
            "clean_train_files",
            "clean_val_only_files",
            "val_overlap_excluded_files",
            "original_train_rows",
            "original_val_rows",
        ],
    )

    # Table 2: generation counts and elapsed time
    combo_counts = Counter((row["platform"], row["adapter"]) for row in manifest)
    combo_elapsed = defaultdict(list)
    for row in manifest:
        combo_elapsed[(row["platform"], row["adapter"])].append(float(row["elapsed_seconds"]))
    generation_rows = []
    for platform in ("shopify", "etsy", "ebay"):
        for adapter in ("ip_adapter", "lora"):
            elapsed = combo_elapsed[(platform, adapter)]
            generation_rows.append(
                {
                    "platform": platform,
                    "adapter": adapter,
                    "outputs": combo_counts[(platform, adapter)],
                    "mean_elapsed_seconds": round(sum(elapsed) / len(elapsed), 3),
                    "median_elapsed_seconds": round(sorted(elapsed)[len(elapsed) // 2], 3),
                }
            )
    write_csv(
        RESULTS_DIR / "sd30_generation_summary.csv",
        generation_rows,
        ["platform", "adapter", "outputs", "mean_elapsed_seconds", "median_elapsed_seconds"],
    )

    # Table 3: run settings
    settings = run_plan["settings"]
    setting_rows = [{"setting": k, "value": v} for k, v in settings.items()]
    write_csv(RESULTS_DIR / "sd30_run_settings_table.csv", setting_rows, ["setting", "value"])

    # Table 4: overfitting summary
    overfit_rows = []
    for platform in ("shopify", "etsy", "ebay"):
        summary = overfit[platform]["summary"]
        overfit_rows.append(
            {
                "platform": platform,
                "best_val_step": summary["best_val_step"],
                "best_val_loss": summary["best_val_loss"],
                "final_val_step": summary["final_val_step"],
                "final_val_loss": summary["final_val_loss"],
                "val_loss_delta_pct": round(summary["val_loss_delta_pct"], 3),
            }
        )
    write_csv(
        RESULTS_DIR / "sd30_overfit_summary.csv",
        overfit_rows,
        [
            "platform",
            "best_val_step",
            "best_val_loss",
            "final_val_step",
            "final_val_loss",
            "val_loss_delta_pct",
        ],
    )

    # Report-ready markdown
    split_md = markdown_table(
        ["Platform", "Clean Train", "Clean Val-Only", "Excluded Val Overlap"],
        [
            [
                row["platform"],
                row["clean_train_files"],
                row["clean_val_only_files"],
                row["val_overlap_excluded_files"],
            ]
            for row in split_rows
        ],
    )
    generation_md = markdown_table(
        ["Platform", "Adapter", "Outputs", "Mean Seconds", "Median Seconds"],
        [
            [
                row["platform"],
                row["adapter"],
                row["outputs"],
                row["mean_elapsed_seconds"],
                row["median_elapsed_seconds"],
            ]
            for row in generation_rows
        ],
    )
    overfit_md = markdown_table(
        ["Platform", "Best Step", "Best Val Loss", "Final Step", "Final Val Loss", "Delta %"],
        [
            [
                row["platform"],
                row["best_val_step"],
                row["best_val_loss"],
                row["final_val_step"],
                row["final_val_loss"],
                row["val_loss_delta_pct"],
            ]
            for row in overfit_rows
        ],
    )

    galleries = [
        "final eval clean val/galleries/overview_first_8_per_combo.jpg",
        "final eval clean val/galleries/shopify_ip_adapter_contact_sheet.jpg",
        "final eval clean val/galleries/shopify_lora_contact_sheet.jpg",
        "final eval clean val/galleries/etsy_ip_adapter_contact_sheet.jpg",
        "final eval clean val/galleries/etsy_lora_contact_sheet.jpg",
        "final eval clean val/galleries/ebay_ip_adapter_contact_sheet.jpg",
        "final eval clean val/galleries/ebay_lora_contact_sheet.jpg",
    ]
    report = f"""# SD-30 Results Tables and Figures

Generated from `final eval clean val/metadata/*` and local SD-21 overfitting outputs.

## Table 1. Clean split sizes

{split_md}

## Table 2. Final evaluation output counts and runtime

{generation_md}

## Table 3. IP-Adapter overfitting summary

{overfit_md}

## Evaluation setup summary

- Total generated outputs: {generation_status["total"]}
- Generated successfully: {generation_status["counts"]["generated"]}
- Adapters evaluated: LoRA and IP-Adapter
- Platforms evaluated: Shopify, Etsy, eBay
- Resolution: {settings["width"]}x{settings["height"]}
- Steps: {settings["steps"]}
- Guidance scale: {settings["guidance_scale"]}
- ControlNet model: `{settings["controlnet_model"]}`
- Adapter scale (all models in clean eval): `0.1`

## Figure assets

"""
    for gallery in galleries:
        report += f"- `{gallery}`\n"

    report += """
## Notes

- This folder is a leakage-free clean validation set built from `data/platform_sets_clean/*/val_only`.
- The clean-eval package contains qualitative outputs and metadata, but not raw reference bundles or metric JSONs for CLIP/FID/LPIPS.
- SD-21 local train/val overfitting analyses are incorporated here through the `results/ip_adapter_*_overfit.json` files.
"""
    (RESULTS_DIR / "sd30_results_tables.md").write_text(report, encoding="utf-8")
    print("Wrote SD-30 tables to results/")


if __name__ == "__main__":
    main()
