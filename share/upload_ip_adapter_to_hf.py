#!/usr/bin/env python
"""
Upload the three trained IP-Adapter checkpoints (plus Etsy's pre-overfit
checkpoint-500) to Hugging Face Hub as a single repo with platform
subdirectories. One-shot idempotent: re-runs only re-upload changed files.

Prerequisites:
    1. An HF account and a "Write"-permission token from
       https://huggingface.co/settings/tokens
    2. `hf auth login` executed in the terminal (token cached to
       ~/.cache/huggingface/token).

Usage:
    python share/upload_ip_adapter_to_hf.py

Edit REPO_ID at the top if forking this work to a different HF user/org.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# Config — edit if re-targeting the upload
# ---------------------------------------------------------------------------

REPO_ID = "jasonshen8848/StudioDiffusion-ip-adapter"
REPO_TYPE = "model"
PRIVATE = False

MODEL_CARD = Path("share/HF_MODEL_CARD.md")
CHECKPOINT_ROOT = Path("checkpoints/ip_adapter")

# Glob patterns relative to CHECKPOINT_ROOT. Only matching files are uploaded.
# Everything else under CHECKPOINT_ROOT (intermediate checkpoints we don't
# want on the Hub) is skipped.
ALLOW_PATTERNS = [
    "shopify/final/*",
    "etsy/final/*",
    "etsy/checkpoint-500/*",   # best Etsy val loss — pre-overfit
    "ebay/final/*",
    "*/train.log",
]

COMMIT_MSG_README = "Add model card"
COMMIT_MSG_WEIGHTS = (
    "Initial release: 3-platform IP-Adapter weights "
    "(Shopify / Etsy / eBay final + Etsy checkpoint-500)"
)


def main() -> None:
    if not MODEL_CARD.exists():
        raise FileNotFoundError(f"Model card not found: {MODEL_CARD}")
    if not CHECKPOINT_ROOT.exists():
        raise FileNotFoundError(
            f"Checkpoint root not found: {CHECKPOINT_ROOT} — "
            "did training complete and are you running from the repo root?"
        )

    api = HfApi()
    print(f"→ Ensuring repo exists: {REPO_ID} (private={PRIVATE})")
    url = create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=PRIVATE,
        exist_ok=True,
    )
    print(f"  repo URL: {url}")

    print(f"→ Uploading README.md (from {MODEL_CARD})")
    api.upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=COMMIT_MSG_README,
    )

    # Size summary for visibility
    def _sz(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 3)

    print(f"→ Uploading checkpoints from {CHECKPOINT_ROOT}")
    for pattern in ALLOW_PATTERNS:
        print(f"    include: {pattern}")
    print(f"  approx source tree size: {_sz(CHECKPOINT_ROOT):.2f} GB (full)")

    api.upload_folder(
        folder_path=str(CHECKPOINT_ROOT),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=ALLOW_PATTERNS,
        commit_message=COMMIT_MSG_WEIGHTS,
    )

    print(f"\n✅ Done. Live at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
