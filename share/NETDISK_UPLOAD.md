# Cloud-drive upload checklist (for the publisher)

## Files to upload

After running `bash share/pack_data_bundle.sh` from the repo, you will get (the date
suffix changes):

1. **`StudioDiffusion-datashare-FULL-YYYYMMDD.tar.gz`** (~500MB+ compressed)  
   - Includes: platform curated sets + manifests, ABO manifest, DeepFashion2 manifest,
     and **all** mask PNGs.  
   - For teammates who need **segmentation evaluation** or **DF2 masks**.

2. **`StudioDiffusion-datashare-PLATFORM-ONLY-YYYYMMDD.tar.gz`** (~70MB)  
   - Only `data/platform_sets/` plus README and snapshot.  
   - For teammates who **only train IP-Adapter / LoRA**, or have limited bandwidth /
     drive quota.

You can upload **both** and label them “full bundle” vs “platform-only” in the drive.

## Suggested text for the drive folder

- Code repo: https://github.com/s-zx/StudioDiffusion  
- Extract **at the repository root** after cloning so `data/platform_sets/` appears.  
- Instructions: **`share/TEAM_DATA_BUNDLE_README.txt`** and **`share/DATA_SNAPSHOT.txt`**
  inside the archive.

## Do not upload / commit

- Do **not** commit `*.tar.gz` into git (they are in `.gitignore`).  
- Do **not** bundle full `data/raw/` into the archive (too large); others should follow
  `docs/data-setup.md` if they need raw sources.
