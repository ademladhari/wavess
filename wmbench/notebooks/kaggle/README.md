# Kaggle wmbench notebooks

One notebook per watermark method. Each runs **attacks one-by-one** (`--resume`), zips `work/<method>/attacked/<attack>/` after each, and uploads to **Hugging Face** for download (Kaggle’s UI download is unreliable).

## Notebooks

| File | Method |
|------|--------|
| `wmbench_kaggle_dct.ipynb` | `dct` |
| `wmbench_kaggle_dwt.ipynb` | `dwt` |
| `wmbench_kaggle_svd.ipynb` | `svd` |
| `wmbench_kaggle_dct_dwt.ipynb` | `dct-dwt` |
| `wmbench_kaggle_dwt_dct_svd.ipynb` | `dwt-dct-svd` |
| `wmbench_kaggle_flexible.ipynb` | `flexible` |
| `wmbench_kaggle_ssl.ipynb` | `ssl` |
| `wmbench_kaggle_robin.ipynb` | `robin` (`--generate-based`) |
| `wmbench_kaggle_tree_ring.ipynb` | `tree-ring` (`--generate-based`) |

Regenerate after editing the template:

```bash
py -3 wmbench/notebooks/generate_kaggle_notebooks.py
```

## Kaggle setup

1. Upload this repo (or clone from GitHub) — notebook expects `/kaggle/working/waves`.
2. **Add Data**:
   - `images500` → set `IMAGES` in CONFIG
   - `negative500` → set `NEGATIVES`
   - Flexible: checkpoint dataset → `WMBENCH_FLEXIBLE_ROOT`
   - ROBIN: optimized watermark checkpoint (`.pt`) → set `WMBENCH_ROBIN_WM_PATH`
3. **Secrets**: `HF_TOKEN` (write token).
4. CONFIG: set `HF_REPO_ID = "youruser/wmbench-exports"` (create empty dataset on HF first).
5. GPU on, Internet on.

## Download flow

After each attack finishes, a zip is written to:

`/kaggle/working/wmbench_exports/<method>/<method>__<attack>.zip`

If `UPLOAD_TO_HF = True`, the notebook prints a **Hugging Face resolve URL** — download on your PC with browser or:

```bash
huggingface-cli download youruser/wmbench-exports wmbench_exports/flexible__Regen-Diff.zip --repo-type dataset --local-dir .
```

**Without Hugging Face:** Save Version on the notebook → on your PC:

```powershell
kaggle kernels output YOUR_USER/wmbench-kaggle-flexible -p D:\downloads\kaggle_out
```

## Attack groups

- **GPU** (default on): `Regen-Diff`, `Regen-VAE`, `Rinse-2xDiff`
- **DIST** (default on): all `Dist-*` and `DistCom-*`
- Toggle `RUN_GPU_ATTACKS` / `RUN_DIST_ATTACKS` in CONFIG.

## OOM

Lower `DIFF_BATCH` to `1` and `LPIPS_BATCH` to `8`.
