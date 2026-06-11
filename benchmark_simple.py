#!/usr/bin/env python3
"""
Simple watermark benchmark for all wmbench methods.

Metrics (per method × attack):
  - Imperceptibility: PSNR, SSIM (host original vs watermarked+attacked)
  - Multi-bit quality: bit accuracy (where the method supports it)
  - Detection: AUROC, TPR @ 1% FPR

Attacks: identity, jpeg_q50, crop, gaussian, regeneration, combined

Methods (via --methods):
  dct | dwt | svd | dct-dwt | dwt-dct-svd | flexible | ssl | robin | tree-ring

Example:
  .venv\\Scripts\\python.exe benchmark_simple.py ^
    --methods dct dwt flexible ^
    --images "D:\\new method\\data\\coco100k\\val" ^
    --n-images 1000 ^
    --output benchmark_simple_output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wmbench.simple_benchmark.runner import run_benchmark  # noqa: E402

ALL_METHODS = (
    "dct",
    "dwt",
    "svd",
    "dct-dwt",
    "dwt-dct-svd",
    "flexible",
    "ssl",
    "robin",
    "tree-ring",
)


def main() -> int:
    p = argparse.ArgumentParser(description="Simple multi-method watermark benchmark")
    p.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help=f"One or more: {' | '.join(ALL_METHODS)}",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path(r"D:\new method\data\coco100k\val"),
    )
    p.add_argument("--output", type=Path, default=_ROOT / "benchmark_simple_output")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed-batch-size", type=int, default=1)
    p.add_argument("--skip-regen", action="store_true", help="Skip regeneration attack")
    p.add_argument(
        "--blind-detect",
        action="store_true",
        help="Force blind detection for methods that support both (e.g. dct, dwt)",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    print(
        f"methods={args.methods} device={device} images={args.images} n={args.n_images}",
        flush=True,
    )

    run_benchmark(
        args.methods,
        args.images,
        args.output,
        n_images=args.n_images,
        image_size=args.image_size,
        device=device,
        skip_regen=args.skip_regen,
        blind_detect=True if args.blind_detect else None,
        embed_batch_size=args.embed_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
