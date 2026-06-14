#!/usr/bin/env python3
"""Load and print ROBIN WAVES benchmark results (results.csv / results.json)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"-Infinity\b", "null", text)
    text = re.sub(r"\bInfinity\b", "null", text)
    return json.loads(text)


def load_results(results_dir: Path) -> tuple[pd.DataFrame, dict | None]:
    results_dir = results_dir.resolve()
    csv_path = results_dir / "results.csv"
    json_path = results_dir / "results.json"

    if csv_path.is_file():
        df = pd.read_csv(csv_path)
    elif json_path.is_file():
        meta = _load_json(json_path)
        df = pd.DataFrame(meta.get("attacks", []))
    else:
        raise FileNotFoundError(f"No results.csv or results.json in {results_dir}")

    meta = _load_json(json_path) if json_path.is_file() else None
    return df, meta


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mean_wm_distance" in out.columns:
        cols = [
            c
            for c in (
                "attack",
                "mean_wm_distance",
                "mean_clean_distance",
                "distance_gap",
                "n_images",
            )
            if c in out.columns
        ]
        formatted = out[cols].copy()
        for col in ("mean_wm_distance", "mean_clean_distance", "distance_gap"):
            if col in formatted.columns:
                formatted[col] = formatted[col].round(1)
        return formatted.rename(
            columns={
                "mean_wm_distance": "wm_L1_dist",
                "mean_clean_distance": "clean_L1_dist",
                "distance_gap": "gap",
            }
        )
    if "PSNR" in out.columns:
        out["PSNR"] = out["PSNR"].replace([float("inf")], float("nan"))
        out["PSNR"] = out["PSNR"].round(1)
    if "SSIM" in out.columns:
        out["SSIM"] = out["SSIM"].round(3)
    for col in ("AUROC", "TPR_at_1pct_FPR"):
        if col in out.columns:
            out[col] = out[col].round(3)
    cols = [c for c in ("attack", "PSNR", "SSIM", "AUROC", "TPR_at_1pct_FPR", "n_images") if c in out.columns]
    return out[cols].rename(columns={"TPR_at_1pct_FPR": "TPR@1%FPR"})


def main() -> int:
    p = argparse.ArgumentParser(description="Decode ROBIN benchmark results.csv/json")
    p.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "outputs_benchmark",
        help="Folder containing results.csv (or results.json)",
    )
    p.add_argument("--csv-out", type=Path, default=None, help="Optional path to write cleaned CSV")
    args = p.parse_args()

    df, meta = load_results(args.results_dir)
    table = format_table(df)

    if meta:
        print("── metadata ──")
        for k in (
            "method",
            "model_id",
            "n_images",
            "seed",
            "wm_path",
            "sanity_identity_distance",
            "mean_clean_distance",
            "sanity_identity_score",
        ):
            if k in meta:
                print(f"  {k}: {meta[k]}")
        print()

    print(table.to_string(index=False))
    print()
    if "AUROC" in table.columns:
        print(f"mean AUROC: {table['AUROC'].mean():.3f}")
    if "TPR@1%FPR" in table.columns:
        print(f"mean TPR@1%FPR: {table['TPR@1%FPR'].mean():.3f}")

    if args.csv_out:
        table.to_csv(args.csv_out, index=False)
        print(f"\nWrote {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
