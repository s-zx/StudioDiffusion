# Data pipeline — completed work (handoff)

This document records **data-related deliverables** implemented in this repo so teammates can see scope, locations, and how to reproduce steps. It intentionally stays out of `README.md`.

For a **short narrative overview** of the same work (motivation, datasets, outcomes, caveats—no runbook), see `docs/data-work-summary.md`.

For **teammate onboarding** (fresh machine, step-by-step commands, disk notes, share-data vs run-scripts), see `docs/data-setup.md`.

---

## Ticket coverage (high level)

| Item | Scope | Status |
|------|--------|--------|
| **SD-8** | Amazon Berkeley Objects (ABO): download strategy + preprocessing | Implemented; sampled download + manifest |
| **SD-6** | DeepFashion2: preprocessing (masks + manifest) | Implemented; assumes raw data present under `data/raw/deepfashion2/` |
| **SD-11** | LAION aesthetics subset | Implemented via public Hugging Face parquet shards + local JPG extraction |
| **SD-16** | Platform-specific image sets (Shopify / Etsy / eBay) | Implemented; CLIP curation + train/val manifests; optional margin / resolution / category-balance flags (see below) |

---

## Repository layout (data)

Raw and processed paths are largely gitignored (large assets). Expected layout after runs:

- **ABO (raw + processed)**  
  - Raw: `data/raw/abo/` (metadata, sampled images per `fetch_abo.py`)  
  - Processed: `data/processed/abo_manifest.csv`

- **DeepFashion2 (raw + processed)**  
  - Raw (actual layout used): `data/raw/deepfashion2/deepfashion2_original_images/{train,validation}/{image,annos}/`  
  - Note: annotations live in **`annos/`**, not `annots/`.  
  - Processed: `data/processed/deepfashion2/masks/{train,validation}/` (PNG masks)  
  - Manifest: `data/processed/deepfashion2_manifest.csv`

- **LAION aesthetics (raw)**  
  - Extracted images: `data/raw/laion_aesthetics/images/` (from parquet in `limingcv/LAION_Aesthetics_512`)

- **Platform sets**  
  - Curated copies: `data/platform_sets/{shopify,etsy,ebay}/`  
  - Splits: `data/platform_sets/manifests/*_train.csv`, `*_val.csv`  
  - Optional CLIP cache (if used): `data/processed/clip_embedding_cache.pt` (also gitignored when present)

---

## Scripts added or materially changed

| Script | Role |
|--------|------|
| `data/fetch_abo.py` | Download ABO metadata + balanced image sample via **boto3** (avoids reliance on `aws` CLI) |
| `data/download_abo.sh` | Shell helper for S3 pulls (options for images / renders) |
| `data/preprocess_abo.py` | Build `abo_manifest.csv`; uses **`images/metadata/images.csv.gz`** for `image_id` → path resolution |
| `data/download_deepfashion2.sh` | Notes + unzip helper for official zips into expected tree |
| `data/preprocess_deepfashion2.py` | Polygon → binary mask PNGs + `deepfashion2_manifest.csv`; **`annos/`** path; default raw root `deepfashion2_original_images/`; **`--num_workers`** for parallel rasterization |
| `data/download_laion.sh` | Pulls parquet shards from `limingcv/LAION_Aesthetics_512`, writes JPGs locally |
| `data/curate_platform.py` | Multi-source CLIP curation; distinctive per-platform prompts; hard assignment (no overlap); optional **`--min_resolution`**, **`--min_margin`**, **`--balance_categories`**; 80/20 manifests; optional embedding cache |

---

## Design decisions worth knowing

1. **ABO image paths**  
   Filenames on disk/S3 are not `main_image_id.jpg`; resolution requires the gzipped **`images.csv.gz`** mapping (handled in `preprocess_abo.py` / ensured by `fetch_abo.py`).

2. **DeepFashion2 layout**  
   The released “original images” pack nests under `deepfashion2_original_images/`; preprocessing defaults to that root.

3. **LAION**  
   Previously attempted HF repos were missing or gated; pipeline uses **embedded images in parquet** from `limingcv/LAION_Aesthetics_512`.

4. **Platform curation**  
   Images are assigned to **one** platform (argmax CLIP score vs archetype text prompts) to avoid duplicate membership across Shopify/Etsy/eBay. Prompts are written for **cross-platform separation** (e.g. pristine white studio vs artisan lifestyle vs utilitarian resale). Optional: **margin filter** drops images where best vs runner-up platform scores differ by less than **`--min_margin`** (reduces ambiguous samples). **`--balance_categories`** uses round‑robin sampling over **`category`** from ABO/DF2 manifests so a single SKU type does not fill the whole set. **`--min_resolution`** (default **224**, short edge) filters images before CLIP; set to **0** to disable.

5. **Apple Silicon**  
   For `curate_platform.py`, **`--device mps`** is much faster than CPU on M-series Macs when CLIP runs locally.

---

## How to re-run (minimal)

From repo root, with project venv activated:

```bash
# ABO: sample + preprocess (adjust args as needed)
python data/fetch_abo.py
python data/preprocess_abo.py

# DeepFashion2: after raw data is in place
python data/preprocess_deepfashion2.py \
  --raw_dir data/raw/deepfashion2/deepfashion2_original_images \
  --num_workers 8

# LAION subset (edit shard counts inside script or extend CLI if added)
bash data/download_laion.sh

# Platform sets — match --sources to your disk (ABO + LAION + DF2, etc.)
python data/curate_platform.py \
  --sources data/raw/abo/images/small \
          data/raw/laion_aesthetics/images \
  --abo_manifest data/processed/abo_manifest.csv \
  --df2_manifest data/processed/deepfashion2_manifest.csv \
  --n_per_platform 400 \
  --min_margin 0.05 \
  --balance_categories \
  --device mps

# Stricter “legacy” behavior (no margin filter, no category balancing, no resolution gate):
#   add the same --sources / manifests / --device as above, then:
#   --min_margin 0 --min_resolution 0   (and do not pass --balance_categories)

python data/curate_platform.py --help   # all flags
```

Exact `--sources` paths depend on your machine; omit manifests you do not have (category columns in CSVs will be empty where no join matches).

---

## Verification already done on author machine

- DeepFashion2: **191,961** train + **32,153** validation annotations processed; **0** skipped; **224,114** manifest rows; masks ~**938 MB** under `data/processed/deepfashion2/`.
- Platform curation run produced **~400** images per platform and train/val CSVs under `data/platform_sets/manifests/` (counts in run log).

Teammates should re-verify paths and row counts after their own downloads, since git does not track large binary outputs.

---

## Open follow-ups (not blocking “data prep” scaffolding)

- **Test split**: Raw tree includes `deepfashion2_original_images/test/`; preprocessing currently targets **`train`** and **`validation`** only. Add `test` to `--splits` if the project needs official test polygons processed the same way.
- **DVC or shared storage**: Large artifacts remain local/gitignored; team may want DVC, rsync, or cloud prefix for shared `data/raw` / `data/processed`.
- **Re-curation**: If sources or prompts change, reuse `--embedding_cache` where applicable to save time. If you change **`--min_resolution`** or **`--sources`**, invalidate or delete the cache so embeddings match the new candidate set.

---

*Last updated: handoff for SD-6 / SD-8 / SD-11 / SD-16 data pipeline implementation.*
