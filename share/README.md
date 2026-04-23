# Share folder (teammate data bundles)

- **`TEAM_DATA_BUNDLE_README.txt`** — Instructions for recipients (also **inside** each `.tar.gz`).
- **`DATA_SNAPSHOT.txt`** — Regenerated when you run the pack script (commit hash, date, contents).
- **`pack_data_bundle.sh`** — Rebuild archives after updating `data/platform_sets` or `data/processed/…`.

## Regenerate bundles

```bash
bash share/pack_data_bundle.sh
```

Produces (date-stamped):

| Archive | Approx. size | Contents |
|---------|----------------|----------|
| `StudioDiffusion-datashare-FULL-*.tar.gz` | ~0.5 GB | `platform_sets` + ABO/DF2 manifests + DF2 masks |
| `StudioDiffusion-datashare-PLATFORM-ONLY-*.tar.gz` | ~70 MB | `platform_sets` + README + snapshot only |

`*.tar.gz` files are **gitignored**; upload them to your cloud drive manually.

## Upload checklist

1. Upload one or both `.tar.gz` files.
2. In the drive folder description, link the repo: `https://github.com/s-zx/StudioDiffusion`
3. Tell teammates: unzip **at repo root**; read **`share/TEAM_DATA_BUNDLE_README.txt`** inside the archive first.

See also **`NETDISK_UPLOAD.md`**.
