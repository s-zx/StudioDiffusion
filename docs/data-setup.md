# Data setup guide (for teammates)

Step-by-step instructions to reproduce the **data pipeline** on a fresh machine. Training/evaluation code expects paths under `data/` as described below.

---

## Should you copy finished data or run the scripts?

| Approach | When to use |
|----------|-------------|
| **Run the scripts yourself** | Default. **Reproducible**: same repo version + same commands → same logic. Good for grading, ablations, and changing curation flags. Requires **time, disk, and network** (especially ABO sample + LAION parquet + DeepFashion2 access). |
| **Receive a shared snapshot** | Use when bandwidth or disk is tight, or you only need to **train/eval** without re-curating. Prefer a **versioned** archive (Drive/S3/lab NAS) with a short **manifest** (row counts, git commit hash, `curate_platform.py` flags). Risk: snapshot can **drift** from `main` if scripts change later. |
| **Recommended team practice** | **Keep scripts as the source of truth** in git. Optionally publish a **pinned** curated bundle (e.g. `platform_sets/` + key CSVs + a `SHA256SUMS` file) for faster onboarding. Anyone who changes curation should **re-run** and update the snapshot note. |

**DeepFashion2** is **gated** (Google Form). Each teammate may need their own approval, or one person shares **only** the processed masks + manifest if license/terms allow—check the dataset license and your course policy.

---

## What to share (processed outputs checklist)

Use this when you upload a **Drive / NAS / tarball** for teammates. The repo includes
`share/pack_data_bundle.sh`, which builds `.tar.gz` archives plus **`share/TEAM_DATA_BUNDLE_README.txt`**
(recipient instructions) and a regenerated **`share/DATA_SNAPSHOT.txt`**. See **`share/NETDISK_UPLOAD.md`**.

Always add a small **`DATA_SNAPSHOT.txt`** (or `README` in the archive) with:

- **Git commit hash** of the repo used to produce the data  
- **Exact command** you ran for `curate_platform.py` (all flags)  
- **Rough sizes** or row counts (optional sanity check)

### Tier 1 — Teammates only train IP-Adapter / LoRA on platform sets

| Share | Why |
|-------|-----|
| **`data/platform_sets/shopify/`**, **`etsy/`**, **`ebay/`** | Copied JPEGs actually used for training (~400 × 3). |
| **`data/platform_sets/manifests/*.csv`** | Train/val splits and **category** metadata. |

**Path caveat:** Manifest **`image_path`** columns still point at **original** locations on *your* machine (e.g. `data/raw/abo/...`), not necessarily at the files under `platform_sets/`. Teammates should **load images from `data/platform_sets/<platform>/`** (by filename or glob), and use the CSV for **split lists + category** only—or you run a one-off script to rewrite `image_path` to repo-relative paths under `platform_sets/` before sharing.

### Tier 2 — Segmentation evaluation (DeepFashion2 IoU vs SAM2, etc.)

| Share | Why |
|-------|-----|
| **`data/processed/deepfashion2_manifest.csv`** | Links each image to its mask path and category. |
| **`data/processed/deepfashion2/masks/`** (train + val as you processed) | Ground-truth PNG masks (~1 GB full). |

They still need the **RGB images** referenced in `image_path`. Options: (a) classmates use their **own** DeepFashion2 download (same layout), or (b) you share a **subset** of `deepfashion2_original_images/.../image/` files that appear in the manifest (smaller than full dataset).

### Tier 3 — ABO-based experiments (same listing metadata as you)

| Share | Why |
|-------|-----|
| **`data/processed/abo_manifest.csv`** | Central index for product type, paths, etc. |
| **`data/raw/abo/`** (at least the **images** + **metadata** your manifest references) | Otherwise `image_path` in the CSV breaks. |

If paths in `abo_manifest.csv` are **absolute**, rewrite to **paths relative to repo root** before sharing, or document the prefix teammates must set.

### Tier 4 — Optional / usually skip

| Item | Note |
|------|------|
| **`data/processed/clip_embedding_cache.pt`** | Speeds **your** re-curation only; large; safe to omit. |
| **Full `data/raw/laion_aesthetics/`** | Not needed if they only use **Tier 1** curated copies. |
| **`.venv`**, **`~/.cache/huggingface/`** | Do not share; everyone builds their own. |

### What *not* to rely on when sharing

- Do not assume classmates have the **same absolute paths** as your laptop. Prefer **relative paths** inside the repo in any CSV you hand off, or document clearly how training code resolves images.

---

## Prerequisites

- **Python 3.11** (matches project tooling).
- **Disk (rough order of magnitude)**  
  - ABO sample (~3K images): a few GB + metadata.  
  - Full DeepFashion2 original images: **tens of GB**.  
  - LAION subset: depends on shard count in `data/download_laion.sh`.  
  - Processed DF2 masks: ~1 GB for full train+val.  
  - Platform sets: small (~400 × 3 + manifests).  
  Plan **≥ 50 GB free** if you want several sources at once.
- **RAM**: preprocessing is mostly CPU-friendly; **CLIP curation** benefits from **GPU** (`cuda`) or **Apple Silicon** (`mps`).

---

## 1. Clone repo and Python environment

```bash
git clone https://github.com/s-zx/StudioDiffusion.git
cd StudioDiffusion
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional dev tools:
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), 'mps', getattr(torch.backends.mps, 'is_available', lambda: False)())"
```

---

## 2. Create expected directories

Git does not store large binaries. Create layout (or let scripts create `processed/` / `platform_sets/`):

```bash
mkdir -p data/raw data/processed data/platform_sets
```

---

## 3. Amazon Berkeley Objects (ABO)

**Goal:** `data/processed/abo_manifest.csv` + sampled images under `data/raw/abo/`.

1. Download metadata + balanced image sample (uses **boto3**, no `aws` CLI required):

   ```bash
   python data/fetch_abo.py --help   # adjust --n_images, output dirs if needed
   ```

2. Build manifest:

   ```bash
   python data/preprocess_abo.py --help
   ```

Ensure `images/metadata/images.csv.gz` is present (fetch script should pull it); manifests **require** it for correct `image_id` → file path resolution.

Shell alternative: `bash data/download_abo.sh` (if `aws` CLI is installed and configured).

---

## 4. DeepFashion2

**Goal:** `data/processed/deepfashion2/masks/{train,validation}/` + `data/processed/deepfashion2_manifest.csv`.

1. **Obtain raw data** via official channel (Google Form). Unpack so images/annotations match:

   `data/raw/deepfashion2/deepfashion2_original_images/{train,validation}/{image,annos}/`

2. See `bash data/download_deepfashion2.sh` for unzip hints.

3. Preprocess (parallel masks):

   ```bash
   python data/preprocess_deepfashion2.py \
     --raw_dir data/raw/deepfashion2/deepfashion2_original_images \
     --out_masks data/processed/deepfashion2/masks \
     --out_csv data/processed/deepfashion2_manifest.csv \
     --splits train validation \
     --num_workers 8
   ```

**Note:** Folder name is **`annos`**, not `annots`. Optional: add `test` to `--splits` if you need the official test split processed the same way.

---

## 5. LAION aesthetics subset

**Goal:** JPEGs under `data/raw/laion_aesthetics/images/` (from Hugging Face parquet).

```bash
bash data/download_laion.sh
```

Edit shard count / limits inside the script if you need a smaller or larger subset. First run may download large parquet files via `huggingface_hub`.

---

## 6. Platform-curated sets (Shopify / Etsy / eBay)

**Goal:** `data/platform_sets/{shopify,etsy,ebay}/` + `data/platform_sets/manifests/*_{train,val}.csv`.

Point `--sources` at **real paths** on your machine (ABO images, LAION folder, DF2 `train/image`, etc.). Pass manifests so CSVs include **category** where possible.

**Recommended** (stricter curation):

```bash
python data/curate_platform.py \
  --sources data/raw/abo/images/small \
          data/raw/laion_aesthetics/images \
          data/raw/deepfashion2/deepfashion2_original_images/train/image \
  --abo_manifest data/processed/abo_manifest.csv \
  --df2_manifest data/processed/deepfashion2_manifest.csv \
  --output data/platform_sets \
  --n_per_platform 400 \
  --min_margin 0.05 \
  --balance_categories \
  --device mps
```

Use `--device cuda` on NVIDIA GPUs. **Legacy-style** run (closest to older behavior): `--min_margin 0 --min_resolution 0` and omit `--balance_categories`.

**Embedding cache:** default `data/processed/clip_embedding_cache.pt` speeds re-runs. Delete it if you change `--sources`, resolution filter, or prompts. Pass `''` to `--embedding_cache` to disable.

Full flag list: `python data/curate_platform.py --help`.

---

## 7. Optional: captions (BLIP-2)

If your training pipeline expects BLIP-2 captions, use `data/generate_captions.py` once images and manifests exist (see script `--help`).

---

## 8. Sanity checks

```bash
wc -l data/processed/abo_manifest.csv
wc -l data/processed/deepfashion2_manifest.csv
ls data/platform_sets/manifests/
head -2 data/platform_sets/manifests/shopify_train.csv
```

---

## 9. Further reading

| Doc | Content |
|-----|--------|
| `docs/team-data-pipeline-handoff.md` | Ticket mapping, design decisions, troubleshooting |
| `docs/data-work-summary.md` | Short narrative of what the data stage does |

---

*If you distribute a tarball of `data/processed/` + `data/platform_sets/`, include a one-line note: git commit hash + exact `curate_platform.py` command used.*
