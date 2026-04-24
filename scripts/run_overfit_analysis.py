#!/usr/bin/env python3
"""Summarize overfitting signals from training logs and optional image metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LINE_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_train_log(path: Path) -> dict[str, list[dict[str, float]]]:
    train_points: list[dict[str, float]] = []
    val_points: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        fields = {k: v for k, v in LINE_RE.findall(raw_line)}
        if "step" not in fields:
            continue
        point = {"step": float(fields["step"])}
        for key in ("train_loss", "val_loss", "lr", "wall", "n"):
            if key in fields:
                try:
                    point[key] = float(fields[key])
                except ValueError:
                    pass
        if "val_loss" in point:
            val_points.append(point)
        elif "train_loss" in point:
            train_points.append(point)
    return {"train": train_points, "val": val_points}


def summarize_overfit(parsed: dict[str, list[dict[str, float]]]) -> dict[str, float | int | None]:
    train_points = parsed["train"]
    val_points = parsed["val"]
    summary: dict[str, float | int | None] = {
        "train_points": len(train_points),
        "val_points": len(val_points),
        "best_val_loss": None,
        "best_val_step": None,
        "final_val_loss": None,
        "final_val_step": None,
        "val_loss_delta": None,
        "val_loss_delta_pct": None,
        "final_train_loss": train_points[-1]["train_loss"] if train_points else None,
    }
    if val_points:
        best = min(val_points, key=lambda p: p["val_loss"])
        final = val_points[-1]
        delta = final["val_loss"] - best["val_loss"]
        delta_pct = (delta / best["val_loss"] * 100.0) if best["val_loss"] else None
        summary.update(
            {
                "best_val_loss": best["val_loss"],
                "best_val_step": int(best["step"]),
                "final_val_loss": final["val_loss"],
                "final_val_step": int(final["step"]),
                "val_loss_delta": delta,
                "val_loss_delta_pct": delta_pct,
            }
        )
    return summary


def collect_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True, help="Path to train.log")
    parser.add_argument("--generated-dir", type=Path, help="Directory of generated images")
    parser.add_argument("--reference-dir", type=Path, help="Directory of held-out real images")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    parsed = parse_train_log(args.log)
    summary = summarize_overfit(parsed)

    if args.generated_dir:
        gen_images = collect_images(args.generated_dir)
        if gen_images:
            from evaluation import CLIPDiversity, FIDScorer

            summary.update(CLIPDiversity(device=args.device).score(gen_images))

            if args.reference_dir:
                ref_images = collect_images(args.reference_dir)
                if ref_images:
                    summary["fid"] = FIDScorer(device=args.device).score(gen_images, ref_images)

    payload = {
        "log_path": str(args.log),
        "summary": summary,
        "curves": parsed,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved overfitting analysis to {args.output}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
