#!/usr/bin/env python3
"""Average per-attack metrics from results_raw.csv → results_averaged.csv."""

from __future__ import annotations

import argparse
import os
import sys

_pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

from wmbench.pipeline.aggregate import write_averaged_from_raw_csv


def main() -> int:
    p = argparse.ArgumentParser(
        description="Average results_raw.csv metrics per (method, attack) across strengths."
    )
    p.add_argument(
        "--raw-csv",
        default="results_raw.csv",
        help="Path to results_raw.csv (default: results_raw.csv in cwd)",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: results_averaged.csv next to raw csv)",
    )
    args = p.parse_args()
    out = write_averaged_from_raw_csv(args.raw_csv, args.output)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
