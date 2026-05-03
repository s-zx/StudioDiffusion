#!/usr/bin/env python3
"""Prepare artifact audits, loss summaries, and pilot sweep configs."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PLATFORMS = ("shopify", "etsy", "ebay")

ADAPTERS: dict[str, dict[str, Any]] = {
    "lora": {
        "config_root": Path("configs/lora"),
        "checkpoint_root": Path("checkpoints/lora"),
        "launcher": "scripts/train_lora.sh",
        "final_files": ("lora_config.json", "pytorch_lora_weights.safetensors"),
    },
    "ip_adapter": {
        "config_root": Path("configs/ip_adapter"),
        "checkpoint_root": Path("checkpoints/ip_adapter"),
        "launcher": "scripts/train_ip_adapter.sh",
        "final_files": ("image_proj_model.pt", "ip_attn_processors.pt"),
    },
}

LINE_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass(frozen=True)
class SweepCandidate:
    adapter: str
    platform: str
    name: str
    priority: int
    changes: dict[str, Any]
    rationale: str
    screening_steps: int | None = None

    @property
    def run_name(self) -> str:
        suffix = f"_s{self.screening_steps}" if self.screening_steps is not None else ""
        return f"{self.platform}_{self.name}{suffix}"

    @property
    def config_path(self) -> Path:
        return Path("configs/tuning") / self.adapter / f"{self.run_name}.yaml"

    @property
    def command(self) -> str:
        launcher = ADAPTERS[self.adapter]["launcher"]
        return f"bash {launcher} {self.config_path}"


def parse_float(value: str) -> float | None:
    cleaned = value.strip().rstrip("s%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_train_log(path: Path) -> dict[str, list[dict[str, float]]]:
    train_points: list[dict[str, float]] = []
    val_points: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = {key: value for key, value in LINE_RE.findall(raw_line)}
        if "step" not in fields:
            continue

        step = parse_float(fields["step"])
        if step is None:
            continue

        point: dict[str, float] = {"step": step}
        for key in ("train_loss", "val_loss", "lr", "wall", "n"):
            if key in fields:
                parsed = parse_float(fields[key])
                if parsed is not None:
                    point[key] = parsed

        if "val_loss" in point:
            val_points.append(point)
        elif "train_loss" in point:
            train_points.append(point)
    return {"train": train_points, "val": val_points}


def unique_by_step(points: list[dict[str, float]]) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    order: list[int] = []
    for point in points:
        step = int(point["step"])
        if step not in by_step:
            order.append(step)
        by_step[step] = point
    return [by_step[step] for step in order]


def summarize_log(path: Path) -> dict[str, Any]:
    parsed = parse_train_log(path)
    train_points = parsed["train"]
    val_points = parsed["val"]
    unique_val_points = unique_by_step(val_points)

    summary: dict[str, Any] = {
        "log_path": str(path),
        "train_points": len(train_points),
        "val_points": len(val_points),
        "unique_val_points": len(unique_val_points),
        "final_train_step": int(train_points[-1]["step"]) if train_points else None,
        "final_train_loss": train_points[-1].get("train_loss") if train_points else None,
        "first_val_step": None,
        "first_val_loss": None,
        "best_val_step": None,
        "best_val_loss": None,
        "final_val_step": None,
        "final_val_loss": None,
        "final_delta_pct_vs_first_val": None,
        "final_minus_best_val_loss": None,
    }
    if unique_val_points:
        first = unique_val_points[0]
        best = min(unique_val_points, key=lambda point: point["val_loss"])
        final = unique_val_points[-1]
        first_loss = first["val_loss"]
        final_loss = final["val_loss"]
        summary.update(
            {
                "first_val_step": int(first["step"]),
                "first_val_loss": first_loss,
                "best_val_step": int(best["step"]),
                "best_val_loss": best["val_loss"],
                "final_val_step": int(final["step"]),
                "final_val_loss": final_loss,
                "final_delta_pct_vs_first_val": (
                    ((final_loss - first_loss) / first_loss) * 100 if first_loss else None
                ),
                "final_minus_best_val_loss": final_loss - best["val_loss"],
            }
        )
    return summary


def load_legacy_lora_summary() -> dict[str, Any]:
    path = Path("results/lora_loss_summary.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("platforms", {})


def load_merged_config(adapter: str, platform: str) -> dict[str, Any] | None:
    config_path = ADAPTERS[adapter]["config_root"] / f"{platform}.yaml"
    if not config_path.exists():
        return None
    cfg = OmegaConf.merge(OmegaConf.load("configs/base.yaml"), OmegaConf.load(config_path))
    return OmegaConf.to_container(cfg, resolve=True)


def artifact_audit() -> dict[str, Any]:
    audit: dict[str, Any] = {
        "date": date.today().isoformat(),
        "adapters": {},
    }
    for adapter, spec in ADAPTERS.items():
        adapter_rows: dict[str, Any] = {}
        for platform in PLATFORMS:
            checkpoint_dir = spec["checkpoint_root"] / platform
            final_dir = checkpoint_dir / "final"
            config_path = spec["config_root"] / f"{platform}.yaml"
            log_path = checkpoint_dir / "train.log"
            final_files = list(spec["final_files"])
            missing_final = [name for name in final_files if not (final_dir / name).exists()]
            cfg = load_merged_config(adapter, platform)
            expected_steps = None
            if cfg:
                expected_steps = int(cfg["training"]["max_train_steps"])

            log_summary = summarize_log(log_path) if log_path.exists() else None
            final_step = log_summary["final_train_step"] if log_summary else None
            log_reaches_expected = (
                expected_steps is not None and final_step is not None and final_step >= expected_steps
            )
            adapter_rows[platform] = {
                "config_path": str(config_path),
                "config_exists": config_path.exists(),
                "checkpoint_dir": str(checkpoint_dir),
                "final_dir": str(final_dir),
                "final_dir_exists": final_dir.exists(),
                "expected_final_files": final_files,
                "missing_final_files": missing_final,
                "train_log": str(log_path),
                "train_log_exists": log_path.exists(),
                "expected_train_steps": expected_steps,
                "final_train_step": final_step,
                "log_reaches_expected_steps": log_reaches_expected,
                "checkpoint_ready": final_dir.exists() and not missing_final,
                "curve_ready": bool(log_summary and log_summary["train_points"] and log_summary["val_points"]),
            }
        audit["adapters"][adapter] = adapter_rows
    return audit


def loss_summary(audit: dict[str, Any]) -> dict[str, Any]:
    legacy_lora = load_legacy_lora_summary()
    payload: dict[str, Any] = {
        "date": date.today().isoformat(),
        "selection_rule": (
            "Rank within the same adapter family and platform by best validation loss, "
            "then final validation loss, stability, earlier best step when tied, and wall-clock."
        ),
        "adapters": {},
    }
    for adapter, spec in ADAPTERS.items():
        platform_rows: dict[str, Any] = {}
        for platform in PLATFORMS:
            row = audit["adapters"][adapter][platform]
            log_path = Path(row["train_log"])
            if log_path.exists():
                platform_rows[platform] = {
                    "status": "from_train_log",
                    **summarize_log(log_path),
                }
            elif adapter == "lora" and platform in legacy_lora:
                platform_rows[platform] = {
                    "status": "legacy_summary_only",
                    "source": legacy_lora[platform].get("source"),
                    "mse_loss": legacy_lora[platform].get("mse_loss"),
                    "rmse": legacy_lora[platform].get("rmse"),
                    "checkpoint_path": legacy_lora[platform].get("checkpoint_path"),
                }
            else:
                platform_rows[platform] = {"status": "missing_train_log"}
        payload["adapters"][adapter] = platform_rows
    return payload


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def sweep_candidates(
    platform: str,
    *,
    screening_steps: int | None = None,
    include_step_candidate: bool = True,
) -> list[SweepCandidate]:
    candidates = [
        SweepCandidate(
            "lora",
            platform,
            "lr5e-5",
            1,
            {"training.learning_rate": 5.0e-5},
            "Lower LR tests whether current LoRA baseline is over-updating.",
            screening_steps,
        ),
        SweepCandidate(
            "lora",
            platform,
            "lr2e-4",
            1,
            {"training.learning_rate": 2.0e-4},
            "Higher LR tests faster convergence versus instability.",
            screening_steps,
        ),
        SweepCandidate(
            "lora",
            platform,
            "rank8_alpha8",
            2,
            {"lora.rank": 8, "lora.alpha": 8},
            "Lower capacity tests whether rank 16 is more than needed.",
            screening_steps,
        ),
        SweepCandidate(
            "lora",
            platform,
            "rank32_alpha32",
            2,
            {"lora.rank": 32, "lora.alpha": 32},
            "Higher capacity tests whether the baseline is underfitting.",
            screening_steps,
        ),
        SweepCandidate(
            "ip_adapter",
            platform,
            "lr5e-5",
            1,
            {"training.learning_rate": 5.0e-5},
            "Lower LR tests whether validation plateaus become smoother.",
            screening_steps,
        ),
        SweepCandidate(
            "ip_adapter",
            platform,
            "lr2e-4",
            1,
            {"training.learning_rate": 2.0e-4},
            "Higher LR tests whether IP-Adapter reaches its plateau earlier.",
            screening_steps,
        ),
        SweepCandidate(
            "ip_adapter",
            platform,
            "tokens8",
            2,
            {"ip_adapter.num_tokens": 8},
            "Fewer image tokens tests a smaller conditioning bottleneck.",
            screening_steps,
        ),
        SweepCandidate(
            "ip_adapter",
            platform,
            "tokens32",
            2,
            {"ip_adapter.num_tokens": 32},
            "More image tokens tests whether extra conditioning capacity helps.",
            screening_steps,
        ),
        SweepCandidate(
            "ip_adapter",
            platform,
            "proj512",
            2,
            {"ip_adapter.proj_hidden_size": 512},
            "Smaller image projection MLP tests whether the baseline projection is over-parameterized.",
            screening_steps,
        ),
        SweepCandidate(
            "ip_adapter",
            platform,
            "proj2048",
            2,
            {"ip_adapter.proj_hidden_size": 2048},
            "Larger image projection MLP tests whether projection capacity is limiting conditioning.",
            screening_steps,
        ),
    ]
    if include_step_candidate:
        candidates.append(
            SweepCandidate(
                "ip_adapter",
                platform,
                "steps2000",
                3,
                {"training.max_train_steps": 2000},
                "Shorter run tests early-stop efficiency from the plateauing baseline.",
                screening_steps,
            )
        )
    return candidates


def speed_probe_candidates(platform: str, steps: int) -> list[SweepCandidate]:
    candidates: list[SweepCandidate] = []
    for adapter in ADAPTERS:
        candidates.extend(
            [
                SweepCandidate(
                    adapter,
                    platform,
                    "baseline",
                    1,
                    {},
                    "Baseline throughput at current 1024px training settings.",
                    steps,
                ),
                SweepCandidate(
                    adapter,
                    platform,
                    "batch4_ga2",
                    1,
                    {
                        "training.train_batch_size": 4,
                        "training.gradient_accumulation_steps": 2,
                    },
                    "Same effective batch as the 2x4 baseline with fewer accumulation cycles.",
                    steps,
                ),
                SweepCandidate(
                    adapter,
                    platform,
                    "no_ckpt",
                    1,
                    {"training.gradient_checkpointing": False},
                    "Tests whether A100 memory can buy faster steps without checkpoint recompute.",
                    steps,
                ),
                SweepCandidate(
                    adapter,
                    platform,
                    "workers4",
                    2,
                    {"training.dataloader_num_workers": 4},
                    "Checks whether image loading is limiting GPU utilization.",
                    steps,
                ),
                SweepCandidate(
                    adapter,
                    platform,
                    "image768",
                    2,
                    {"data.image_size": 768},
                    "Measures the speed/quality tradeoff of a lower training resolution.",
                    steps,
                ),
            ]
        )
    return candidates


def quality_screen_candidates(platforms: tuple[str, ...], steps: int) -> list[SweepCandidate]:
    candidates: list[SweepCandidate] = []
    for platform in platforms:
        candidates.extend(
            [
                SweepCandidate(
                    "lora",
                    platform,
                    "baseline",
                    1,
                    {},
                    "250-step LoRA baseline for same-step candidate comparison.",
                    steps,
                ),
                SweepCandidate(
                    "lora",
                    platform,
                    "lr5e-5",
                    1,
                    {"training.learning_rate": 5.0e-5},
                    "Lower LR tests whether current LoRA baseline is over-updating.",
                    steps,
                ),
                SweepCandidate(
                    "lora",
                    platform,
                    "lr2e-4",
                    1,
                    {"training.learning_rate": 2.0e-4},
                    "Higher LR tests faster convergence versus instability.",
                    steps,
                ),
                SweepCandidate(
                    "lora",
                    platform,
                    "rank8_alpha8",
                    2,
                    {"lora.rank": 8, "lora.alpha": 8},
                    "Lower capacity tests whether rank 16 is more than needed.",
                    steps,
                ),
                SweepCandidate(
                    "lora",
                    platform,
                    "rank32_alpha16",
                    2,
                    {"lora.rank": 32, "lora.alpha": 16},
                    "Higher rank with baseline-scale strength separates capacity from adapter scale.",
                    steps,
                ),
                SweepCandidate(
                    "lora",
                    platform,
                    "rank16_alpha32",
                    2,
                    {"lora.rank": 16, "lora.alpha": 32},
                    "Same capacity with stronger LoRA scaling tests adapter strength.",
                    steps,
                ),
                SweepCandidate(
                    "lora",
                    platform,
                    "dropout0.05",
                    2,
                    {"lora.dropout": 0.05},
                    "Small dropout tests regularization on the small platform datasets.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "baseline",
                    1,
                    {},
                    "250-step IP-Adapter baseline with scheduler and min-SNR parity.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "lr5e-5",
                    1,
                    {"training.learning_rate": 5.0e-5},
                    "Lower LR tests whether validation plateaus become smoother.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "lr2e-4",
                    1,
                    {"training.learning_rate": 2.0e-4},
                    "Higher LR tests whether IP-Adapter reaches its plateau earlier.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "tokens4",
                    2,
                    {"ip_adapter.num_tokens": 4},
                    "Very small image-token bottleneck tests speed and over-conditioning risk.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "tokens8",
                    2,
                    {"ip_adapter.num_tokens": 8},
                    "Fewer image tokens tests a smaller conditioning bottleneck.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "tokens32",
                    2,
                    {"ip_adapter.num_tokens": 32},
                    "More image tokens tests whether extra conditioning capacity helps.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "projlinear",
                    2,
                    {"ip_adapter.proj_hidden_size": None},
                    "Original linear projection baseline tests whether the MLP is helping.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "proj512",
                    2,
                    {"ip_adapter.proj_hidden_size": 512},
                    "Smaller projection MLP tests whether the baseline projection is over-parameterized.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "proj2048",
                    2,
                    {"ip_adapter.proj_hidden_size": 2048},
                    "Larger projection MLP tests whether projection capacity is limiting conditioning.",
                    steps,
                ),
                SweepCandidate(
                    "ip_adapter",
                    platform,
                    "snr0",
                    3,
                    {"noise.snr_gamma": 0.0},
                    "Compares the previous plain-MSE IP objective against min-SNR weighting.",
                    steps,
                ),
            ]
        )
    return candidates


def pilot_candidates(platform: str) -> list[SweepCandidate]:
    return sweep_candidates(platform)


def screening_candidates(platforms: tuple[str, ...], steps: int) -> list[SweepCandidate]:
    return [
        candidate
        for platform in platforms
        for candidate in sweep_candidates(
            platform,
            screening_steps=steps,
            include_step_candidate=False,
        )
    ]


def create_candidate_config(candidate: SweepCandidate) -> dict[str, Any]:
    source_path = ADAPTERS[candidate.adapter]["config_root"] / f"{candidate.platform}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(source_path), resolve=False)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping config: {source_path}")

    config = deepcopy(config)
    config["platform"] = candidate.run_name
    config["description"] = (
        f"Tuning candidate for {candidate.adapter}/{candidate.platform}: "
        f"{candidate.name}. {candidate.rationale}"
    )
    config["tuning"] = {
        "base_platform": candidate.platform,
        "adapter": candidate.adapter,
        "candidate": candidate.name,
        "run_name": candidate.run_name,
        "priority": candidate.priority,
        "rationale": candidate.rationale,
        "changes": candidate.changes,
        "screening_steps": candidate.screening_steps,
        "selection_metric": "best validation loss within adapter/platform",
    }
    if candidate.screening_steps is not None:
        set_nested(config, "training.max_train_steps", candidate.screening_steps)
        set_nested(config, "training.validation_steps", candidate.screening_steps)
        set_nested(config, "training.checkpointing_steps", candidate.screening_steps)
        set_nested(config, "lr_scheduler.num_warmup_steps", min(50, max(0, candidate.screening_steps // 10)))
    for dotted_key, value in candidate.changes.items():
        set_nested(config, dotted_key, value)
    return config


def write_candidate_configs(candidates: list[SweepCandidate]) -> None:
    for candidate in candidates:
        candidate.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = create_candidate_config(candidate)
        OmegaConf.save(OmegaConf.create(config), candidate.config_path)


def sweep_plan(platform_label: str, candidates: list[SweepCandidate]) -> dict[str, Any]:
    return {
        "date": date.today().isoformat(),
        "pilot_platform": platform_label,
        "strategy": "Use train/validation loss to shortlist SD-17 candidates before image eval.",
        "ranking": [
            "best validation loss within the same adapter/platform",
            "final validation loss",
            "stable train/validation curve",
            "earlier best step when tied",
            "wall-clock efficiency",
        ],
        "candidates": [
            {
                "adapter": candidate.adapter,
                "platform": candidate.platform,
                "name": candidate.name,
                "run_name": candidate.run_name,
                "priority": candidate.priority,
                "changes": candidate.changes,
                "screening_steps": candidate.screening_steps,
                "rationale": candidate.rationale,
                "config_path": str(candidate.config_path),
                "command": candidate.command,
            }
            for candidate in candidates
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_artifact_audit_md(path: Path, audit: dict[str, Any]) -> None:
    lines = ["# Adapter Tuning Artifact Audit", "", f"Date: {audit['date']}", ""]
    lines.append(
        "| Adapter | Platform | Config | Checkpoint Ready | Curve Ready | Final Step | Missing Final Files |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for adapter in ADAPTERS:
        for platform in PLATFORMS:
            row = audit["adapters"][adapter][platform]
            missing = ", ".join(row["missing_final_files"]) or "none"
            lines.append(
                f"| {adapter} | {platform} | {yes_no(row['config_exists'])} | "
                f"{yes_no(row['checkpoint_ready'])} | {yes_no(row['curve_ready'])} | "
                f"{row['final_train_step'] or 'n/a'} | {missing} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_loss_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Adapter Tuning Loss Summary", "", f"Date: {summary['date']}", ""]
    lines.append(summary["selection_rule"])
    lines.append("")
    for adapter in ADAPTERS:
        lines.append(f"## {adapter}")
        lines.append("")
        lines.append(
            "| Platform | Status | Final Train | First Val | Final Val | Best Val | Best Step | Delta vs First | Points |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for platform in PLATFORMS:
            row = summary["adapters"][adapter][platform]
            if row["status"] == "from_train_log":
                lines.append(
                    f"| {platform} | {row['status']} | {fmt(row['final_train_loss'])} | "
                    f"{fmt(row['first_val_loss'])} | {fmt(row['final_val_loss'])} | "
                    f"{fmt(row['best_val_loss'])} | {row['best_val_step']} | "
                    f"{fmt_pct(row['final_delta_pct_vs_first_val'])} | "
                    f"{row['train_points']}/{row['unique_val_points']} |"
                )
            elif row["status"] == "legacy_summary_only":
                lines.append(
                    f"| {platform} | legacy_summary_only | n/a | n/a | "
                    f"{fmt(row.get('mse_loss'))} | n/a | n/a | n/a | n/a |"
                )
            else:
                lines.append(f"| {platform} | {row['status']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sweep_plan_md(path: Path, plan: dict[str, Any]) -> None:
    lines = ["# Adapter Tuning Sweep Plan", "", f"Date: {plan['date']}", ""]
    lines.append(f"Pilot platform: `{plan['pilot_platform']}`")
    lines.append("")
    lines.append(plan["strategy"])
    lines.append("")
    lines.append("Ranking rule:")
    lines.append("")
    for index, rule in enumerate(plan["ranking"], start=1):
        lines.append(f"{index}. {rule}")
    lines.append("")
    lines.append("| Priority | Adapter | Candidate | Changes | Config | Command |")
    lines.append("|---:|---|---|---|---|---|")
    for candidate in plan["candidates"]:
        changes = ", ".join(f"{key}={value}" for key, value in candidate["changes"].items())
        lines.append(
            f"| {candidate['priority']} | {candidate['adapter']} | {candidate['name']} | "
            f"{changes} | `{candidate['config_path']}` | `{candidate['command']}` |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}%"


def with_extra_changes(candidate: SweepCandidate, extra_changes: dict[str, Any]) -> SweepCandidate:
    return SweepCandidate(
        adapter=candidate.adapter,
        platform=candidate.platform,
        name=candidate.name,
        priority=candidate.priority,
        changes={**extra_changes, **candidate.changes},
        rationale=candidate.rationale,
        screening_steps=candidate.screening_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-platform",
        choices=PLATFORMS,
        default="shopify",
        help="Platform used for the first small sweep config set.",
    )
    parser.add_argument(
        "--all-platforms",
        action="store_true",
        help="Generate sweep configs for Shopify, Etsy, and eBay.",
    )
    parser.add_argument(
        "--screening-steps",
        type=int,
        help="Set max_train_steps for every generated candidate and omit step-count candidates.",
    )
    parser.add_argument(
        "--plan",
        choices=("legacy", "speed", "quality"),
        default="legacy",
        help="Generate the legacy sweep, the 100-step speed probe, or the 250-step quality screen.",
    )
    parser.add_argument(
        "--skip-configs",
        action="store_true",
        help="Write audits and summaries without generating tuning config files.",
    )
    args = parser.parse_args()

    results_dir = Path("results")
    audit = artifact_audit()
    summary = loss_summary(audit)
    if args.plan == "speed":
        steps = args.screening_steps or 100
        platform_label = args.pilot_platform
        candidates = speed_probe_candidates(args.pilot_platform, steps)
    elif args.plan == "quality":
        steps = args.screening_steps or 250
        platforms = PLATFORMS if args.all_platforms else (args.pilot_platform,)
        platform_label = "all" if args.all_platforms else args.pilot_platform
        candidates = quality_screen_candidates(platforms, steps)
        candidates = [
            with_extra_changes(candidate, {"training.gradient_checkpointing": False})
            for candidate in candidates
        ]
    elif args.all_platforms:
        platform_label = "all"
        if args.screening_steps is None:
            candidates = [candidate for platform in PLATFORMS for candidate in pilot_candidates(platform)]
        else:
            candidates = screening_candidates(PLATFORMS, args.screening_steps)
    else:
        platform_label = args.pilot_platform
        if args.screening_steps is None:
            candidates = pilot_candidates(args.pilot_platform)
        else:
            candidates = screening_candidates((args.pilot_platform,), args.screening_steps)
    plan = sweep_plan(platform_label, candidates)
    plan["plan"] = args.plan

    if not args.skip_configs:
        write_candidate_configs(candidates)

    write_json(results_dir / "adapter_tuning_artifact_audit.json", audit)
    write_json(results_dir / "adapter_tuning_loss_summary.json", summary)
    write_json(results_dir / "adapter_tuning_sweep_plan.json", plan)
    write_artifact_audit_md(results_dir / "adapter_tuning_artifact_audit.md", audit)
    write_loss_summary_md(results_dir / "adapter_tuning_loss_summary.md", summary)
    write_sweep_plan_md(results_dir / "adapter_tuning_sweep_plan.md", plan)
    if args.plan != "legacy":
        write_json(results_dir / f"adapter_tuning_{args.plan}_plan.json", plan)
        write_sweep_plan_md(results_dir / f"adapter_tuning_{args.plan}_plan.md", plan)

    print("Wrote tuning prep outputs:")
    print("  results/adapter_tuning_artifact_audit.{json,md}")
    print("  results/adapter_tuning_loss_summary.{json,md}")
    print("  results/adapter_tuning_sweep_plan.{json,md}")
    if args.plan != "legacy":
        print(f"  results/adapter_tuning_{args.plan}_plan.{{json,md}}")
    if not args.skip_configs:
        suffix = f"_s{args.screening_steps}" if args.screening_steps is not None else ""
        platform_label = "*" if args.all_platforms else args.pilot_platform
        print(f"  configs/tuning/*/{platform_label}_*{suffix}.yaml")


if __name__ == "__main__":
    main()