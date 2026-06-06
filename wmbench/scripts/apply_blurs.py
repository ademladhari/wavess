from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageFilter


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Gaussian blur radii to an image and save results.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("--out-dir", type=str, default="blurs", help="Output directory (created if missing)")
    parser.add_argument("--start", type=float, default=0.0, help="Start radius (inclusive)")
    parser.add_argument("--end", type=float, default=20.0, help="End radius (inclusive)")
    parser.add_argument("--step", type=float, default=1.0, help="Step between radii")
    parser.add_argument("--format", type=str, default="png", help="Output image format/extension")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(inp).convert("RGB")
    stem = inp.stem

    # Generate radii inclusive of end
    r = args.start
    radii = []
    # Guard against floating-point accumulation by computing count
    if args.step <= 0:
        raise SystemExit("--step must be > 0")
    count = int(((args.end - args.start) / args.step) + 1)
    for i in range(max(0, count)):
        val = args.start + i * args.step
        if val > args.end + 1e-8:
            break
        radii.append(round(val, 6))

    for radius in radii:
        out_name = f"{stem}_blur_{radius}.{args.format}"
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing: {out_path}")
            continue
        if radius == 0:
            out_img = img.copy()
        else:
            out_img = img.filter(ImageFilter.GaussianBlur(radius))
        out_img.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
