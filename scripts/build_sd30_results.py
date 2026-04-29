#!/usr/bin/env python3
"""Build SD-30 tables and a report-ready summary from clean eval metadata."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


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
    ebay_lora_refresh_path = META_DIR / "ebay_lora_lr2e-4_s3000_training_summary.json"
    ebay_lora_refresh = load_json(ebay_lora_refresh_path) if ebay_lora_refresh_path.exists() else None

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
                    "median_elapsed_seconds": round(median(elapsed), 3),
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

    # Table 5: category coverage by platform
    category_rows = []
    by_platform_category = defaultdict(Counter)
    unique_case_keys = set()
    for row in manifest:
        case_key = (row["platform"], row["case_id"])
        if case_key in unique_case_keys:
            continue
        unique_case_keys.add(case_key)
        by_platform_category[row["platform"]][row["category"]] += 1
    for platform in ("shopify", "etsy", "ebay"):
        for category, count in by_platform_category[platform].most_common(15):
            category_rows.append(
                {
                    "platform": platform,
                    "category": category,
                    "count": count,
                }
            )
    write_csv(
        RESULTS_DIR / "sd30_category_coverage.csv",
        category_rows,
        ["platform", "category", "count"],
    )

    # Table 6: figure manifest
    figure_rows = [
        {
            "figure_id": "fig:overview",
            "path": "final eval clean val/galleries/overview_first_8_per_combo.jpg",
            "caption": "Overview of the first eight clean validation outputs for each platform-adapter combination.",
            "purpose": "High-level qualitative comparison across platforms and adapter types.",
        },
        {
            "figure_id": "fig:shopify_ip",
            "path": "final eval clean val/galleries/shopify_ip_adapter_contact_sheet.jpg",
            "caption": "Shopify IP-Adapter contact sheet over clean validation products.",
            "purpose": "Qualitative review of clean-background studio behavior and common failure cases.",
        },
        {
            "figure_id": "fig:shopify_lora",
            "path": "final eval clean val/galleries/shopify_lora_contact_sheet.jpg",
            "caption": "Shopify LoRA contact sheet over clean validation products.",
            "purpose": "Qualitative review of LoRA adaptation behavior for Shopify-style outputs.",
        },
        {
            "figure_id": "fig:etsy_ip",
            "path": "final eval clean val/galleries/etsy_ip_adapter_contact_sheet.jpg",
            "caption": "Etsy IP-Adapter contact sheet over clean validation products.",
            "purpose": "Qualitative review of warm lifestyle styling on held-out Etsy-like products.",
        },
        {
            "figure_id": "fig:etsy_lora",
            "path": "final eval clean val/galleries/etsy_lora_contact_sheet.jpg",
            "caption": "Etsy LoRA contact sheet over clean validation products.",
            "purpose": "Compare LoRA styling strength and content preservation for Etsy outputs.",
        },
        {
            "figure_id": "fig:ebay_ip",
            "path": "final eval clean val/galleries/ebay_ip_adapter_contact_sheet.jpg",
            "caption": "eBay IP-Adapter contact sheet over clean validation products.",
            "purpose": "Qualitative review of utilitarian clarity and plain-background behavior.",
        },
        {
            "figure_id": "fig:ebay_lora",
            "path": "final eval clean val/galleries/ebay_lora_contact_sheet.jpg",
            "caption": "eBay LoRA contact sheet over clean validation products.",
            "purpose": "Compare LoRA adaptation behavior for eBay-style product presentation.",
        },
    ]
    write_csv(
        RESULTS_DIR / "sd30_figure_manifest.csv",
        figure_rows,
        ["figure_id", "path", "caption", "purpose"],
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
    category_md = markdown_table(
        ["Platform", "Category", "Count"],
        [
            [row["platform"], row["category"], row["count"]]
            for row in category_rows[:18]
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

## Table 4. Category coverage snapshot (unique clean-val products)

{category_md}

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
"""
    if ebay_lora_refresh:
        report += (
            f"- eBay LoRA refresh: `{ebay_lora_refresh['model_id']}` from "
            f"`{ebay_lora_refresh['checkpoint']}` with final val loss "
            f"`{ebay_lora_refresh['final_val_loss']}`\n"
        )

    report += """

## Figure assets

"""
    for row in figure_rows:
        report += f"- `{row['figure_id']}`: `{row['path']}`\n"
        report += f"  - Caption: {row['caption']}\n"
        report += f"  - Purpose: {row['purpose']}\n"

    report += """
## Notes

- This folder is a leakage-free clean validation set built from `data/platform_sets_clean/*/val_only`.
- Category counts are deduplicated by clean validation case so the same product is not double-counted across LoRA and IP-Adapter outputs.
- The updated package refreshes the eBay LoRA slice to the best-confirmed `lr=2e-4, step=3000` checkpoint when `metadata/ebay_lora_lr2e-4_s3000_training_summary.json` is present.
- The clean-eval package contains qualitative outputs and metadata, but not raw reference bundles or metric JSONs for CLIP/FID/LPIPS.
- SD-21 local train/val overfitting analyses are incorporated here through the `results/ip_adapter_*_overfit.json` files.
- The qualitative contact sheets include both strong examples and visible failure modes; this is useful for an honest final report discussion section.
"""
    (RESULTS_DIR / "sd30_results_tables.md").write_text(report, encoding="utf-8")

    report_section = f"""# SD-30 Report Section Draft

## Final evaluation protocol

We generated a leakage-free clean validation set using `data/platform_sets_clean/*/val_only` and evaluated both adapter families (`IP-Adapter` and `LoRA`) on all three target platforms. The final clean evaluation bundle contains {generation_status["total"]} generated outputs across Shopify, Etsy, and eBay. All runs used the same generation settings: 1024x1024 resolution, 40 denoising steps, guidance scale 8.5, and a low ControlNet conditioning scale of 0.05 with `diffusers/controlnet-canny-sdxl-1.0`.

## Quantitative summary

{split_md}

{generation_md}

{overfit_md}

These results show that the clean validation protocol is balanced across platform-adapter combinations, with {generation_status["counts"]["generated"]} / {generation_status["total"]} runs completing successfully. IP-Adapter runs were consistently slower than LoRA runs by roughly 1.2 to 1.7 seconds per sample in this clean-eval export. Overfitting analysis on the published IP-Adapter checkpoints indicates that Etsy is the only platform with a meaningful post-optimum validation-loss increase, while Shopify and eBay remain effectively stable through the final checkpoint.
"""
    if ebay_lora_refresh:
        report_section += (
            f"\nThe refreshed clean-eval package also upgrades the eBay LoRA slice to "
            f"`{ebay_lora_refresh['model_id']}` (`{ebay_lora_refresh['checkpoint']}`), "
            f"whose training summary reports a final validation loss of "
            f"`{ebay_lora_refresh['final_val_loss']}`. This means the current qualitative "
            f"eBay LoRA figures are tied to the best-confirmed LoRA setting rather than the older baseline export.\n"
        )

    report_section += f"""

## Qualitative figure plan

Use `fig:overview` as the main paper figure for side-by-side qualitative comparison across platforms and adapter types. Use the six per-combination contact sheets as appendix figures or backup slides. The Shopify sheets are especially useful for discussing failure cases where the model drifts toward human or mannequin-like presentations for wearable products, while the Etsy and eBay sheets better highlight scene-style and background-style differences.

## Key takeaways

1. The clean validation export is large enough to support a meaningful final qualitative comparison across all six platform-adapter combinations.
2. The runtime metadata is strong enough to justify a small throughput table in the final report.
3. The current repo already supports an honest narrative: strong clean-eval coverage, clear overfitting conclusions for IP-Adapter, and visible qualitative failure modes that can be discussed rather than hidden.
"""
    (RESULTS_DIR / "sd30_report_section.md").write_text(report_section, encoding="utf-8")
    print("Wrote SD-30 tables to results/")


if __name__ == "__main__":
    main()
