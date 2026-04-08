# Data pipeline work — overview

Short description of **what the dataset stage covers today**: sources, outcomes, and caveats. Step-by-step paths and commands are in `docs/team-data-pipeline-handoff.md`.

---

## Motivation

StudioDiffusion targets **platform-specific product aesthetics**. This work assembles **catalog/fashion imagery**, **masks where needed**, and **aesthetic-diverse pools** into manifests and curated subsets via scripts (large files stay local or on shared storage, not in git).

---

## Datasets in scope

| Dataset | Role |
|--------|------|
| **Amazon Berkeley Objects (ABO)** | E‑commerce-style product photos and metadata; main “real product” source. |
| **DeepFashion2** | Fashion images with instance segmentation → we build **binary masks** + manifest for conditioning workflows. |
| **LAION Aesthetics (HF parquet mirror)** | Subset of aesthetic images **decoded from parquet** to JPEG for curation and diversity. |
| **Curated platform sets** | **Shopify / Etsy / eBay** buckets from CLIP-based assignment + train/val CSVs (not a new official benchmark). |

---

## What was implemented (summary)

- **ABO:** Metadata + balanced image sampling; **CSV manifest** with correct **image ID → path** resolution via official `images.csv.gz`.
- **DeepFashion2:** Polygons rasterized to **one mask per image**; **CSV manifest** (image, mask, category, split).
- **LAION subset:** Parquet shards → **local JPEGs** for downstream use.
- **Platform curation:** CLIP scores vs **distinct per-platform text archetypes** (stylized prompts to separate Shopify / Etsy / eBay), **hard assignment** (one platform per image), top‑K per site, **80/20 splits**. Optional improvements: **`--min_resolution`** (drop tiny/corrupt images before embedding), **`--min_margin`** (drop images where the top two platform scores are too close—ambiguous style), **`--balance_categories`** (round‑robin category sampling so one product type does not dominate a platform set). Defaults preserve backward compatibility except **`--min_resolution`** which defaults to **224**; use **`0`** to skip the resolution filter.

---

## Artifacts

- Manifests for ABO and DeepFashion2.
- PNG mask trees for DeepFashion2 (train/validation).
- Curated platform image dirs + split manifests under `data/platform_sets/`.
- Scripts under `data/` (see handoff doc).

---

## Caveats

Workloads are **subset-sized** for storage and iteration. **Platform tags** are **CLIP-defined** (weak labels for style experiments). **DeepFashion2** preprocessing ran on **train + validation**; **test** can be added the same way if needed. After changing **`--sources`**, resolution filter, or CLIP prompts, **delete or bypass** `data/processed/clip_embedding_cache.pt` so embeddings are recomputed on the correct image list.

---

*Detail: `docs/team-data-pipeline-handoff.md`.*
