#!/usr/bin/env python3
"""Resize a folder of images to a uniform square size for wmbench benchmarks.

Default (center-crop + 512x512 LANCZOS) matches Colab ``images512`` / tree-ring /
flexible expectations and avoids mixed portrait/landscape sizes breaking batched metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _list_images(directory: str) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")
    paths = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    if not paths:
        raise FileNotFoundError(f"No images found under {directory}")
    return paths


def resize_image(
    image: Image.Image,
    *,
    size: int,
    mode: str,
) -> Image.Image:
    im = image.convert("RGB")
    w, h = im.size
    if mode == "stretch":
        return im.resize((size, size), Image.Resampling.LANCZOS)
    if mode == "center_crop_square":
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        return im.resize((size, size), Image.Resampling.LANCZOS)
    if mode == "longest_edge":
        scale = size / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return im.resize((nw, nh), Image.Resampling.LANCZOS)
    raise ValueError(f"Unknown resize mode: {mode!r}")


def prepare_folder(
    src_dir: str,
    dst_dir: str,
    *,
    size: int,
    mode: str,
    overwrite: bool,
) -> dict:
    src_paths = _list_images(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for src in tqdm(src_paths, desc=f"prepare/{dst.name}", unit="img"):
        out = dst / src.name
        if out.is_file() and not overwrite:
            continue
        with Image.open(src) as im:
            orig_size = im.size
            out_im = resize_image(im, size=size, mode=mode)
        out_im.save(out, quality=95)
        manifest.append(
            {
                "filename": src.name,
                "source": str(src.resolve()),
                "original_size": [int(orig_size[0]), int(orig_size[1])],
                "output_size": [int(out_im.size[0]), int(out_im.size[1])],
            }
        )

    meta = {
        "source_dir": str(Path(src_dir).resolve()),
        "output_dir": str(dst.resolve()),
        "size": int(size),
        "mode": mode,
        "count": len(manifest),
        "files": manifest,
    }
    with open(dst / ".wmbench_prepare_manifest.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare uniform-size image folders for wmbench.")
    p.add_argument("--src", required=True, help="Source image directory")
    p.add_argument(
        "--dst",
        default=None,
        help="Output directory (default: <src>512 e.g. D:\\dataset\\images -> images512)",
    )
    p.add_argument("--size", type=int, default=512, help="Square output side length (default: 512)")
    p.add_argument(
        "--mode",
        choices=("center_crop_square", "stretch", "longest_edge"),
        default="center_crop_square",
        help="Resize policy (default: center_crop_square, matches tree-ring/colab images512)",
    )
    p.add_argument("--overwrite", action="store_true", help="Rewrite existing outputs")
    args = p.parse_args(argv)

    src = os.path.abspath(args.src)
    if args.dst:
        dst = os.path.abspath(args.dst)
    else:
        base = Path(src)
        dst = str(base.parent / f"{base.name}{int(args.size)}")

    meta = prepare_folder(src, dst, size=int(args.size), mode=str(args.mode), overwrite=bool(args.overwrite))
    print(
        f"Prepared {meta['count']} images -> {meta['output_dir']} "
        f"({meta['mode']}, {meta['size']}x{meta['size']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
