#!/usr/bin/env python3
"""Generate one Kaggle .ipynb per wmbench method."""

from __future__ import annotations

import json
from pathlib import Path

METHODS: dict[str, dict] = {
    "dct": {
        "title": "wmbench — DCT",
        "requirements": "requirements_colab_combined.txt",
        "generate_based": False,
        "extra_env_cell": None,
    },
    "dwt": {
        "title": "wmbench — DWT",
        "requirements": "requirements_colab_combined.txt",
        "generate_based": False,
        "extra_env_cell": None,
    },
    "svd": {
        "title": "wmbench — SVD",
        "requirements": "requirements_colab_combined.txt",
        "generate_based": False,
        "extra_env_cell": None,
    },
    "dct-dwt": {
        "title": "wmbench — DCT-DWT",
        "requirements": "requirements_colab_combined.txt",
        "generate_based": False,
        "extra_env_cell": None,
    },
    "dwt-dct-svd": {
        "title": "wmbench — DWT-DCT-SVD",
        "requirements": "requirements_colab_combined.txt",
        "generate_based": False,
        "extra_env_cell": None,
    },
    "flexible": {
        "title": "wmbench — Flexible",
        "requirements": "requirements_colab_wmbench_flexible.txt",
        "generate_based": False,
        "extra_env_cell": """# Flexible checkpoints (attach Kaggle dataset or unzip under /kaggle/working)
import os
from pathlib import Path

FLEX_ROOT = Path(os.environ.get("WMBENCH_FLEXIBLE_ROOT", "/kaggle/working/moe_stabilized_v2_milder_epoch200"))
os.environ["WMBENCH_FLEXIBLE_ROOT"] = str(FLEX_ROOT)
os.environ["WMBENCH_FLEXIBLE_CHECKPOINTS"] = str(FLEX_ROOT / "checkpoints")
os.environ["WMBENCH_FLEXIBLE_PROMPT"] = os.environ.get("WMBENCH_FLEXIBLE_PROMPT", "a photo")
print("FLEX_ROOT:", FLEX_ROOT, "exists:", FLEX_ROOT.exists())""",
    },
    "ssl": {
        "title": "wmbench — SSL",
        "requirements": "requirements_colab_wmbench_flexible.txt",
        "generate_based": False,
        "extra_env_cell": """# SSL weights (attach dataset with encoder/decoder ckpts)
import os
from pathlib import Path

SSL_ROOT = Path("/kaggle/working/waves/ssl")
os.environ["WMBENCH_SSL_ROOT"] = str(SSL_ROOT)
print("SSL_ROOT:", SSL_ROOT, "exists:", SSL_ROOT.exists())""",
    },
    "tree-ring": {
        "title": "wmbench — Tree-Ring",
        "requirements": "requirements_colab_wmbench_flexible.txt",
        "generate_based": True,
        "extra_env_cell": """# Tree-Ring code + SD weights
import os
from pathlib import Path

os.environ.setdefault("WMBENCH_TREE_RING_ROOT", "/kaggle/working/waves/tree-ring")
print("WMBENCH_TREE_RING_ROOT:", os.environ["WMBENCH_TREE_RING_ROOT"])""",
    },
}


def _cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in source.splitlines()],
    }


def build_notebook(method: str, cfg: dict) -> dict:
    gen_based = cfg["generate_based"]
    req_file = cfg["requirements"]

    config_py = f'''# === CONFIG (edit paths) ===
METHOD = "{method}"
WAVES_ROOT = "/kaggle/working/waves"          # clone target
IMAGES = "/kaggle/input/images500/images"     # host images (not needed if GENERATE_BASED)
NEGATIVES = "/kaggle/input/negative500"       # clean negatives
OUTPUT_DIR = f"/kaggle/working/wmbench_results/{{METHOD}}"
EXPORT_DIR = f"/kaggle/working/wmbench_exports/{{METHOD}}"

# Attacks: run GPU set, DIST set, or ALL
RUN_GPU_ATTACKS = True
RUN_DIST_ATTACKS = True
SKIP_RINSE4X = True
# Per-attack benchmark + zip + optional HF upload (checkpointed, reliable).

# Download: upload each zip to Hugging Face (set HF_TOKEN in Kaggle Secrets)
UPLOAD_TO_HF = True
HF_REPO_ID = "YOUR_HF_USERNAME/wmbench-exports"  # dataset repo

GENERATE_BASED = {gen_based}
DIFF_BATCH = 4
LPIPS_BATCH = 16
GENERATED_COUNT = 64   # tree-ring / generate-based only

# Skip CLIP-H aesthetics load (removes 396-line "Loading weights" spam; faster on Kaggle)
SKIP_AESTHETICS = True
'''

    install_py = '''import os, sys, subprocess
from pathlib import Path

# Clone waves repo (use your fork if private)
WAVES_ROOT = Path("/kaggle/working/waves")
if not (WAVES_ROOT / "wmbench" / "run_benchmark.py").exists():
    # Option A: git clone (edit YOUR_USER). Option B: Add Data -> upload waves zip/dataset
    !git clone --recurse-submodules --depth 1 https://github.com/YOUR_USER/waves.git /kaggle/working/waves

os.chdir(WAVES_ROOT)
print("WAVES_ROOT:", WAVES_ROOT.resolve())

# PyTorch CUDA (adjust cu124 if needed)
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
!pip install -q -r wmbench/''' + req_file + '''
!pip install -q git+https://github.com/openai/CLIP.git huggingface_hub

sys.path.insert(0, str(WAVES_ROOT))

# Fewer duplicate tqdm lines in Kaggle
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
try:
    from transformers.utils import logging as _tf_log
    _tf_log.disable_progress_bar()
except Exception:
    pass

# Hugging Face token from Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("HF_TOKEN", UserSecretsClient().get_secret("HF_TOKEN"))
except Exception:
    pass
'''

    load_common_py = '''def _load_kaggle_common(waves_root):
    """Import kaggle_wmbench_common.py (works even if wmbench.notebooks is not a package)."""
    import importlib.util
    p = Path(waves_root) / "wmbench" / "notebooks" / "kaggle_wmbench_common.py"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Upload the full waves repo (must include wmbench/notebooks/) "
            "or copy kaggle_wmbench_common.py into the notebook."
        )
    spec = importlib.util.spec_from_file_location("kaggle_wmbench_common", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
'''

    run_py = f'''import os
import sys
from pathlib import Path

# Load config from previous cell
assert "METHOD" in dir(), "Run the config cell first."
WAVES_ROOT = Path(WAVES_ROOT)
sys.path.insert(0, str(WAVES_ROOT))

{load_common_py}
_common = _load_kaggle_common(WAVES_ROOT)
GPU_ATTACKS = _common.GPU_ATTACKS
DIST_ATTACKS = _common.DIST_ATTACKS
run_per_attack_with_zips = _common.run_per_attack_with_zips
zip_full_results = _common.zip_full_results
upload_to_huggingface = _common.upload_to_huggingface

attacks = []
if RUN_GPU_ATTACKS:
    attacks.extend(GPU_ATTACKS)
if RUN_DIST_ATTACKS:
    attacks.extend(DIST_ATTACKS)
if SKIP_RINSE4X:
    attacks = [a for a in attacks if a != "Rinse-4xDiff"]

out = Path(OUTPUT_DIR)
exp = Path(EXPORT_DIR)
exp.mkdir(parents=True, exist_ok=True)

images = None if GENERATE_BASED else IMAGES
negatives = None if GENERATE_BASED else NEGATIVES

extra = []
if GENERATE_BASED:
    extra = ["--generated-count", str(GENERATED_COUNT)]

entries = run_per_attack_with_zips(
    method=METHOD,
    waves_root=WAVES_ROOT,
    output_dir=out,
    attack_names=attacks,
    export_dir=exp,
    hf_repo_id=HF_REPO_ID if UPLOAD_TO_HF else None,
    upload_each=UPLOAD_TO_HF,
    images=images,
    negatives=negatives,
    generate_based=GENERATE_BASED,
    diff_batch=DIFF_BATCH,
    lpips_batch=LPIPS_BATCH,
    skip_rinse4x=SKIP_RINSE4X,
    skip_aesthetics=SKIP_AESTHETICS,
    extra_argv=extra,
)

print("\\nDone per-attack zips:", len(entries))
for e in entries:
    print(e.get("hf_url") or e["zip"])

# Also build/upload a full bundle from this same cell (metrics + csv + plots).
full_zip = zip_full_results(out, exp, METHOD)
print("\\nFull results zip:", full_zip, f"({{full_zip.stat().st_size/1e9:.3f}} GB)")
if UPLOAD_TO_HF:
    full_url = upload_to_huggingface(full_zip, HF_REPO_ID)
    print("Full results HF:", full_url)
'''

    final_py = '''# Final aggregate zip (CSVs + plots + full work/)
from pathlib import Path

_common = _load_kaggle_common(WAVES_ROOT)
full_zip = _common.zip_full_results(Path(OUTPUT_DIR), Path(EXPORT_DIR), METHOD)
print("Full results zip:", full_zip, f"({full_zip.stat().st_size/1e9:.3f} GB)")

if UPLOAD_TO_HF:
    url = _common.upload_to_huggingface(full_zip, HF_REPO_ID)
    print("Download full bundle:", url)

# List all exports in /kaggle/working (use HF links above, or Save Version -> Output tab)
for z in sorted(Path(EXPORT_DIR).glob("*.zip")):
    print(z, z.stat().st_size / 1e6, "MB")
'''

    md = f"""# {cfg['title']} — Kaggle benchmark

Runs **one attack at a time** with `--resume`, then zips each attack folder to HF with attacked images, detection scores, **metrics** (LPIPS, FID, aesthetics, etc.), and CSV snapshots. Uploads `{method}__FULL_RESULTS.zip` at the end.

## Before you run
1. **Add Data**: `images500`, `negative500` (and flexible/ssl checkpoints if needed).
2. **Secrets**: `HF_TOKEN` (Hugging Face write token) for downloads.
3. Edit **CONFIG** cell: dataset paths, `HF_REPO_ID`, git clone URL.
4. **GPU**: T4 x2 or P100 recommended for Regen attacks; use `DIFF_BATCH=1` if OOM.

## Download results
Kaggle's Output download button is unreliable. This notebook uploads each zip to **Hugging Face** — use the printed `https://huggingface.co/datasets/...` links.

Alternatively: **Save Version** on this notebook, then on your PC:
`kaggle kernels output USER/THIS_NOTEBOOK -p ./out`
"""

    cells = [
        _cell("markdown", md),
        _cell("code", config_py),
        _cell("code", install_py),
    ]
    if cfg.get("extra_env_cell"):
        cells.append(_cell("code", cfg["extra_env_cell"]))
    cells.extend(
        [
            _cell("code", run_py),
            _cell("code", final_py),
        ]
    )

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isGpuEnabled": True},
        },
        "cells": cells,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, cfg in METHODS.items():
        nb = build_notebook(method, cfg)
        path = out_dir / f"wmbench_kaggle_{method.replace('-', '_')}.ipynb"
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("Wrote", path)


if __name__ == "__main__":
    main()
