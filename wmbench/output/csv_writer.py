from __future__ import annotations

import os


def write_results_raw(output_dir: str, raw_rows: list[dict]) -> None:
    """Compatibility wrapper; prefer pipeline.aggregate._write_results_raw."""
    import csv

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results_raw.csv")
    if not raw_rows:
        return
    fieldnames = [
        "method",
        "attack",
        "strength",
        "P",
        "Q",
        "PSNR",
        "SSIM",
        "NMI",
        "LPIPS",
        "FID",
        "CLIP_FID",
        "aesthetics_delta",
        "artifacts",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)


def write_results_leaderboard(output_dir: str, rows: list[dict]) -> None:
    import csv

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results_leaderboard.csv")
    fieldnames = ["method", "attack", "rank", "Q@0.7P", "Q@0.4P", "Avg_P", "Avg_Q"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
