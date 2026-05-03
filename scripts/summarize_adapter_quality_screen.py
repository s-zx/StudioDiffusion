#!/usr/bin/env python3
"""Summarize adapter quality-screen sweep results."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


LINE_RE = re.compile(r"(\w+)=([^\s]+)")
START_RE = re.compile(r"^START adapter=(\S+) run=(\S+) date=(\S+)")
DONE_RE = re.compile(r"^DONE adapter=(\S+) run=(\S+) date=(\S+)")
FAILED_RE = re.compile(r"^FAILED adapter=(\S+) run=(\S+) date=(\S+)(?: exit_code=(\S+))?")
SUMMARY_RE = re.compile(r"^SUMMARY adapter=(\S+) run=(\S+) (.*)$")


def parse_fields(text: str) -> dict[str, str]:
    return {key: value for key, value in LINE_RE.findall(text)}


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.rstrip("s"))
    except ValueError:
        return None


def parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def load_config(adapter: str, run: str) -> dict[str, Any]:
    path = Path("configs/tuning") / adapter / f"{run}.yaml"
    if not path.exists():
        return {"config_path": str(path), "config_exists": False}
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping config: {path}")
    tuning = raw.get("tuning", {}) or {}
    training = raw.get("training", {}) or {}
    adapter_cfg = raw.get("lora" if adapter == "lora" else "ip_adapter", {}) or {}
    return {
        "config_path": str(path),
        "config_exists": True,
        "platform": tuning.get("base_platform") or raw.get("platform", "").split("_")[0],
        "candidate": tuning.get("candidate") or run,
        "changes": tuning.get("changes", {}) or {},
        "learning_rate": training.get("learning_rate"),
        "gradient_checkpointing": training.get("gradient_checkpointing"),
        "max_train_steps": training.get("max_train_steps"),
        "train_batch_size": training.get("train_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "rank": adapter_cfg.get("rank"),
        "alpha": adapter_cfg.get("alpha"),
        "dropout": adapter_cfg.get("dropout"),
        "num_tokens": adapter_cfg.get("num_tokens"),
        "proj_hidden_size": adapter_cfg.get("proj_hidden_size"),
    }


def parse_stage_log(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        start = START_RE.match(line)
        if start:
            adapter, run, date = start.groups()
            rows.setdefault((adapter, run), {}).update({"adapter": adapter, "run": run, "start": date})
            continue

        done = DONE_RE.match(line)
        if done:
            adapter, run, date = done.groups()
            rows.setdefault((adapter, run), {}).update({"adapter": adapter, "run": run, "done": date})
            continue

        failed = FAILED_RE.match(line)
        if failed:
            adapter, run, date, exit_code = failed.groups()
            rows.setdefault((adapter, run), {}).update(
                {"adapter": adapter, "run": run, "failed": date, "exit_code": exit_code}
            )
            continue

        summary = SUMMARY_RE.match(line)
        if summary:
            adapter, run, fields_text = summary.groups()
            fields = parse_fields(fields_text)
            rows.setdefault((adapter, run), {}).update(
                {
                    "adapter": adapter,
                    "run": run,
                    "step": int(parse_float(fields.get("step")) or 0),
                    "val_loss": parse_float(fields.get("val_loss")),
                    "validation_wall_seconds": parse_float(fields.get("wall")),
                    "validation_batches": int(parse_float(fields.get("n")) or 0),
                }
            )
    return rows


def enrich_rows(rows_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (adapter, run), row in sorted(rows_by_key.items()):
        cfg = load_config(adapter, run)
        start = parse_datetime(row.get("start"))
        done = parse_datetime(row.get("done"))
        duration_seconds = (done - start).total_seconds() if start and done else None
        checkpoint_path = Path("checkpoints") / adapter / run / "final"
        rows.append(
            {
                **row,
                **cfg,
                "status": "failed" if "failed" in row else "done" if "done" in row else "unknown",
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60 if duration_seconds is not None else None,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.exists(),
            }
        )
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["adapter"], row["platform"])].append(row)

    ranked: list[dict[str, Any]] = []
    for key, group in grouped.items():
        baseline = next((row for row in group if row.get("candidate") == "baseline"), None)
        baseline_loss = baseline.get("val_loss") if baseline else None
        sorted_group = sorted(
            group,
            key=lambda row: (
                row.get("val_loss") is None,
                row.get("val_loss") if row.get("val_loss") is not None else float("inf"),
                row.get("duration_seconds") or float("inf"),
            ),
        )
        for index, row in enumerate(sorted_group, start=1):
            val_loss = row.get("val_loss")
            delta = val_loss - baseline_loss if baseline_loss is not None and val_loss is not None else None
            pct = (delta / baseline_loss) * 100 if baseline_loss and delta is not None else None
            ranked.append({**row, "rank_within_adapter_platform": index, "delta_vs_baseline": delta, "pct_vs_baseline": pct})
    return sorted(ranked, key=lambda row: (row["adapter"], row["platform"], row["rank_within_adapter_platform"]))


def build_payload(rows: list[dict[str, Any]], log_path: Path) -> dict[str, Any]:
    winners = [row for row in rows if row["rank_within_adapter_platform"] == 1]
    return {
        "stage": "adapter_quality_screen_250",
        "progress_log": str(log_path),
        "selection_rule": "Rank only within the same adapter family and platform by lowest validation loss.",
        "run_count": len(rows),
        "failure_count": sum(1 for row in rows if row["status"] == "failed"),
        "checkpoint_count": sum(1 for row in rows if row["checkpoint_exists"]),
        "average_duration_minutes": mean([row["duration_minutes"] for row in rows if row.get("duration_minutes") is not None]),
        "average_duration_minutes_by_adapter": {
            adapter: mean([row["duration_minutes"] for row in rows if row["adapter"] == adapter and row.get("duration_minutes") is not None])
            for adapter in sorted({row["adapter"] for row in rows})
        },
        "winners": winners,
        "runs": rows,
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Adapter Quality Screen 250 Summary", ""]
    lines.append(f"Progress log: `{payload['progress_log']}`")
    lines.append("")
    lines.append(f"Runs: {payload['run_count']} complete, {payload['failure_count']} failed, {payload['checkpoint_count']} final checkpoints.")
    lines.append(f"Average runtime: {fmt(payload['average_duration_minutes'], 2)} min/run.")
    lines.append("")
    lines.append("## Winners")
    lines.append("")
    lines.append("| Adapter | Platform | Candidate | Val Loss | Delta vs Baseline | Runtime Min | Checkpoint |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in payload["winners"]:
        lines.append(
            f"| {row['adapter']} | {row['platform']} | {row['candidate']} | {fmt(row['val_loss'])} | "
            f"{fmt(row['delta_vs_baseline'])} ({fmt(row['pct_vs_baseline'], 3)}%) | "
            f"{fmt(row['duration_minutes'], 2)} | `{row['checkpoint_path']}` |"
        )
    lines.append("")
    for adapter in sorted({row["adapter"] for row in payload["runs"]}):
        lines.append(f"## {adapter}")
        lines.append("")
        for platform in sorted({row["platform"] for row in payload["runs"] if row["adapter"] == adapter}):
            lines.append(f"### {platform}")
            lines.append("")
            lines.append("| Rank | Candidate | Val Loss | Delta vs Baseline | Runtime Min | Key Changes |")
            lines.append("|---:|---|---:|---:|---:|---|")
            group = [row for row in payload["runs"] if row["adapter"] == adapter and row["platform"] == platform]
            for row in sorted(group, key=lambda item: item["rank_within_adapter_platform"]):
                changes = ", ".join(f"{key}={value}" for key, value in row.get("changes", {}).items()) or "baseline"
                lines.append(
                    f"| {row['rank_within_adapter_platform']} | {row['candidate']} | {fmt(row['val_loss'])} | "
                    f"{fmt(row['delta_vs_baseline'])} ({fmt(row['pct_vs_baseline'], 3)}%) | "
                    f"{fmt(row['duration_minutes'], 2)} | {changes} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="results/adapter_quality_screen_250_live_progress.log")
    parser.add_argument("--json", default="results/adapter_quality_screen_250_summary.json")
    parser.add_argument("--md", default="results/adapter_quality_screen_250_summary.md")
    args = parser.parse_args()

    log_path = Path(args.log)
    rows = rank_rows(enrich_rows(parse_stage_log(log_path)))
    payload = build_payload(rows, log_path)
    write_json(Path(args.json), payload)
    write_markdown(Path(args.md), payload)
    print(f"wrote {args.json}")
    print(f"wrote {args.md}")
    print(f"runs={payload['run_count']} failures={payload['failure_count']} checkpoints={payload['checkpoint_count']}")


if __name__ == "__main__":
    main()