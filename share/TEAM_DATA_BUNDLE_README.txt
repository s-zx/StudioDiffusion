================================================================================
StudioDiffusion — Teammate data bundle (cloud / drive share)
================================================================================

WHAT THIS IS
------------
This archive contains **processed** project data. After you extract it at the **repository
root**, it merges with the cloned repo’s `data/` tree so you can train platform adapters
(IP-Adapter / LoRA) or run DeepFashion2 segmentation evaluation **without** redoing full
downloads and preprocessing (saves time and bandwidth).

WHERE TO EXTRACT
----------------
1. Clone: https://github.com/s-zx/StudioDiffusion
2. `cd` into the repo root (same level as `README.md` and `data/`).
3. Extract this archive here with **merge / replace** so you end up with:

   <repo-root>/
     share/
       TEAM_DATA_BUNDLE_README.txt   ← this file (paths may vary by bundle layout)
       DATA_SNAPSHOT.txt              ← version / inventory (if included under share/)
     data/
       platform_sets/                 ← Shopify / Etsy / eBay curated JPEGs + manifests
       processed/
         abo_manifest.csv
         deepfashion2_manifest.csv
         deepfashion2/masks/          ← train/val binary mask PNGs

If the tarball already contains a top-level `data/` folder, extract at the repo root so
final paths are `<repo-root>/data/...`.

PLATFORM SETS (platform_sets)
-----------------------------
- `data/platform_sets/{shopify,etsy,ebay}/` — JPEG copies used for training (~400 per
  platform, approximate).
- `data/platform_sets/manifests/*_train.csv` and `*_val.csv` — 80/20 splits and a
  **category** column.

NOTE — `image_path` in the CSVs is the **packager's absolute path** (e.g.
`/Users/mac/.../data/raw/abo/...`) and will not exist on your computer. The
manifest-driven `ProductDataset` (`adapters/ip_adapter/train.py`) handles this
automatically: it derives the on-disk filename from each row's `image_path` using
the `curate_platform.py` output naming rule (`<parent>_<basename>` first, plain
`<basename>` as fallback) and looks the file up under
`data/platform_sets/<platform>/`. You do **not** need to rewrite paths or build
a custom loader — `bash scripts/train_ip_adapter.sh <platform>` works directly.

If you write your own loader (e.g. for the LoRA track that hasn't migrated yet),
either reuse `ProductDataset(platform_dir, split, image_size)` or apply the same
basename-resolution rule yourself.

ABO / DEEPFASHION2 MANIFESTS AND MASKS
--------------------------------------
- `abo_manifest.csv` — ABO sample index and paths (still useful as metadata if you do
  not have the matching raw images locally).
- `deepfashion2_manifest.csv` + `deepfashion2/masks/` — polygon annotations rasterized
  to binary PNGs for official train/val.

For IoU-style evaluation you still need the **RGB images** referenced by `image_path`
in the DeepFashion2 manifest: either download DeepFashion2 yourself with the same
layout, or obtain a subset from the team. See `docs/data-setup.md`.

NOT INCLUDED
------------
- Full `data/raw/` trees (ABO bulk, extracted LAION images, etc.) — too large.
- `clip_embedding_cache.pt` — only speeds repeated CLIP curation; can be regenerated.

FURTHER READING
---------------
- `docs/data-setup.md` — full reproducible pipeline from scratch
- `docs/team-data-pipeline-handoff.md` — scripts and design notes

LICENSING AND COURSE POLICY
---------------------------
ABO, DeepFashion2, LAION, etc. each have their own licenses; DeepFashion2 is **gated**.
Follow your course rules and dataset terms. Do not republish this bundle in ways that
violate those licenses.

================================================================================
