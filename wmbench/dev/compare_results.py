from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any


NUMERIC_KEYS = (
    "P",
    "Q",
    "Q@0.7P",
    "Q@0.4P",
    "PSNR",
    "SSIM",
    "NMI",
    "LPIPS",
    "FID",
    "CLIP_FID",
    "aesthetics_delta",
    "artifacts",
)


def _load_csv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("method", ""),
        row.get("attack", ""),
        row.get("strength", ""),
    )


def _to_float(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare baseline vs optimized wmbench CSV outputs.")
    p.add_argument("--baseline", required=True, help="Path to baseline results_raw.csv")
    p.add_argument("--candidate", required=True, help="Path to optimized results_raw.csv")
    p.add_argument("--tol", type=float, default=1e-4, help="Absolute tolerance for numeric drift")
    args = p.parse_args(argv)

    base_rows = _load_csv(args.baseline)
    cand_rows = _load_csv(args.candidate)
    base_map = {_key(r): r for r in base_rows}
    cand_map = {_key(r): r for r in cand_rows}
    missing = sorted(set(base_map) - set(cand_map))
    extra = sorted(set(cand_map) - set(base_map))
    if missing:
        print(f"Missing rows in candidate: {len(missing)}")
        for k in missing[:10]:
            print(f"  {k}")
        return 2
    if extra:
        print(f"Extra rows in candidate: {len(extra)}")
        for k in extra[:10]:
            print(f"  {k}")
        return 2

    worst_key = ""
    worst = 0.0
    failures: list[str] = []
    for key, b_row in base_map.items():
        c_row = cand_map[key]
        for col in NUMERIC_KEYS:
            if col not in b_row or col not in c_row:
                continue
            b = _to_float(b_row.get(col))
            c = _to_float(c_row.get(col))
            if math.isnan(b) and math.isnan(c):
                continue
            diff = abs(b - c)
            if diff > worst:
                worst = diff
                worst_key = f"{key}::{col}"
            if diff > args.tol:
                failures.append(f"{key}::{col} baseline={b} candidate={c} diff={diff}")

    print(
        f"Compared {len(base_map)} rows from {os.path.basename(args.baseline)} vs "
        f"{os.path.basename(args.candidate)}"
    )
    print(f"Worst drift: {worst:.6g} at {worst_key or 'n/a'}")
    if failures:
        print(f"FAILED: {len(failures)} values exceed tol={args.tol}")
        for line in failures[:20]:
            print("  " + line)
        return 1
    print(f"PASS: all compared values within tol={args.tol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
