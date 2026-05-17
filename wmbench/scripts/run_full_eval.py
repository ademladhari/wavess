#!/usr/bin/env python3
"""Run full evaluation: decode all, then compute metrics for all modes.

This is a convenience wrapper that calls the existing `decode` and `metric`
scripts as subprocesses in the current virtualenv/python.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

METRIC_MODES = [
    "psnr",
    "ssim",
    "nmi",
    "lpips",
    "watson",
    "aesthetics_and_artifacts",
    "clip_score",
    "legacy_fid",
    "clean_fid",
    "clip_fid",
]


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


@click.command()
@click.option("--data-dir", required=True, help="Dataset root containing attack folders")
@click.option("--result-dir", required=True, help="Result root for JSON outputs")
@click.option("--subset", is_flag=True, default=False, help="Run on subset only")
@click.option("--limit", type=int, default=None, help="Image count limit")
@click.option("--quiet", is_flag=True, default=False, help="Quiet mode")
@click.option("--skip-clip-score", is_flag=True, default=False, help="Skip clip_score metric")
@click.option("--skip-watson", is_flag=True, default=False, help="Skip watson metric")
@click.option("--skip-fid", is_flag=True, default=False, help="Skip all *fid metrics")
def main(
    data_dir: str,
    result_dir: str,
    subset: bool,
    limit: int | None,
    quiet: bool,
    skip_clip_score: bool,
    skip_watson: bool,
    skip_fid: bool,
) -> None:
    python = sys.executable

    # Decode all image dirs under data_dir
    decode_cmd = [python, "-m", "wmbench.scripts.decode", "--path", data_dir, "--all", "--include-clean"]
    if subset:
        decode_cmd.append("--subset")
    if quiet:
        decode_cmd.append("--quiet")
    _run(decode_cmd)

    # Run metrics for each mode
    prompts_path = Path(result_dir) / "mscoco" / "prompts.json"
    for mode in METRIC_MODES:
        if mode == "watson" and skip_watson:
            print("Skipping watson (flagged)")
            continue
        if mode.endswith("_fid") and skip_fid:
            print(f"Skipping {mode} (flagged)")
            continue
        if mode == "clip_score":
            if skip_clip_score:
                print("Skipping clip_score (flagged)")
                continue
            if not prompts_path.exists():
                print(f"Skipping clip_score (missing prompts): {prompts_path}")
                continue
        metric_cmd = [python, "-m", "wmbench.scripts.metric", mode, "--data-dir", data_dir, "--result-dir", result_dir]
        if subset:
            metric_cmd.append("--subset")
        if limit is not None:
            metric_cmd.extend(["--limit", str(limit)])
        if quiet:
            metric_cmd.append("--quiet")
        _run(metric_cmd)


if __name__ == "__main__":
    main()
