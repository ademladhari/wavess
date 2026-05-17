#!/usr/bin/env python3
"""Embed DCT watermarks into images without importing from waves/.

Generates DCT-embedded image sets ready for downstream attacks and detection.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import click
import numpy as np
from PIL import Image

import dotenv

dotenv.load_dotenv(override=False)

from wmbench.dev import (
    LIMIT,
    SUBSET_LIMIT,
    DATASET_NAMES,
)

import subprocess
from pathlib import Path
import sys


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(list(DATASET_NAMES.keys())),
    required=True,
    help="Dataset split to process (diffusiondb, mscoco, dalle3).",
)
@click.option("--data-dir", type=click.Path(exists=False), default=None, help="Override DATA_DIR for this run.")
@click.option("--limit", type=int, default=None, help="Limit number of images (sets WAVES_LIMIT).")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing DCT images.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode.")
def _invoke_inplace_dct_script(dataset: str, data_dir: str | None, limit: int | None, overwrite: bool, quiet: bool) -> None:
    """Invoke the original dct method script in-place (dct/reproduce_dct_paper.py).

    This runs the upstream DCT embedding in the `dct/` folder so all method logic and
    parameters remain inside the `dct/` repo as you requested.
    """
    repo_root = Path(__file__).resolve().parents[2]
    dct_script = repo_root / "dct" / "reproduce_dct_paper.py"
    if not dct_script.exists():
        raise RuntimeError(
            f"Expected upstream dct script not found: {dct_script}.\nRun DCT embedding in the dct/ folder directly."
        )

    # Prepare environment for subprocess
    env = os.environ.copy()
    if data_dir is not None:
        env["DATA_DIR"] = str(data_dir)
    # If a numeric limit is provided, expose it to the upstream script via WAVES_LIMIT
    if limit is not None:
        env["WAVES_LIMIT"] = str(limit)
        env["WAVES_SUBSET_LIMIT"] = str(limit)

    # Upstream script expects --dataset-dir
    cmd = [str(dct_script), "--dataset-dir", env.get("DATA_DIR", "")]
    # Run using same Python executable; pass modified env
    subprocess.check_call([sys.executable] + cmd, cwd=str(dct_script.parent), env=env)


def main() -> None:
    """Entry point for module execution — delegate to Click command."""
    # Call the Click-decorated function which will parse sys.argv
    _invoke_inplace_dct_script()


if __name__ == "__main__":
    main()
