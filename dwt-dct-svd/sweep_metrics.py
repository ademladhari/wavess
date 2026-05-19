"""
Grid search α, β for PSNR (Sec. 4.1) and NCC (Eq. 7) — same pipeline as run_watermark.py.

Writes a CSV of all runs, then prints which (α, β) achieve:
  - highest PSNR (imperceptibility),
  - highest NCC(TW) vs TW_extracted,
  - highest NCC(signature) vs extracted (biometric resolution),
  plus optional balanced score and top-5 tables.

Example:
  .\\.venv\\Scripts\\python.exe sweep_metrics.py --cover-size 512
  .\\.venv\\Scripts\\python.exe sweep_metrics.py --alphas "0.02,0.04,0.06" --betas "0.001,0.002,0.004"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from skimage import color, io, util
from skimage.transform import resize

from dwt_dct_svd import (
    embed_signature_in_biometric,
    embed_tw_in_cover,
    extract_signature_from_tw,
    extract_tw_from_watermarked,
    ncc,
    psnr_channelwise,
    to_float01,
)
from run_watermark import _find_biometric_image, _find_image, _rgba_to_rgb_u8


def _run_one(
    cover,
    biometric,
    signature,
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    TW, state = embed_signature_in_biometric(biometric, signature, alpha=alpha)
    watermarked, state = embed_tw_in_cover(cover, TW, beta=beta, state=state)
    psnr_m, _ = psnr_channelwise(cover, watermarked)
    TW_ext = extract_tw_from_watermarked(watermarked, cover, beta=beta, state=state)
    sig_ext = extract_signature_from_tw(TW_ext, biometric, alpha=alpha, state=state)
    tw_ncc = ncc(TW, TW_ext)
    sig_ncc = ncc(signature, sig_ext)
    return psnr_m, tw_ncc, sig_ncc


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Sweep α, β; report best PSNR / best NCC pairs and save CSV."
    )
    ap.add_argument("--cover-size", type=int, default=512, help="Resize cover to S×S (use None via 0 to skip)")
    ap.add_argument("--bio-size", type=int, default=256)
    ap.add_argument("--sig-size", type=int, default=128)
    ap.add_argument("--csv", type=Path, default=root / "output" / "sweep_metrics.csv")
    ap.add_argument(
        "--alphas",
        type=str,
        default="0.02,0.04,0.06,0.08,0.10",
        help="Comma-separated α list",
    )
    ap.add_argument(
        "--betas",
        type=str,
        default="0.0005,0.001,0.002,0.004,0.006,0.008,0.01",
        help="Comma-separated β list",
    )
    ap.add_argument(
        "--w-psnr",
        type=float,
        default=1.0,
        help="Weight for PSNR in balanced score (PSNR/70 + w_ncc*NCC_tw)",
    )
    ap.add_argument("--w-ncc-tw", type=float, default=1.0, help="Weight for NCC(TW) in balanced score")
    args = ap.parse_args()

    orig_path = _find_image(root, "original")
    pic_path = _find_biometric_image(root)
    logo_path = _find_image(root, "logo")

    cover = _rgba_to_rgb_u8(io.imread(orig_path))
    if args.cover_size and int(args.cover_size) > 0:
        s = int(args.cover_size)
        cover = resize(to_float01(cover), (s, s), anti_aliasing=True)
    picture_u8 = _rgba_to_rgb_u8(io.imread(pic_path))
    logo_u8 = _rgba_to_rgb_u8(io.imread(logo_path))

    bio_gray = color.rgb2gray(to_float01(picture_u8))
    biometric = resize(bio_gray, (args.bio_size, args.bio_size), anti_aliasing=True)
    sig_gray = color.rgb2gray(to_float01(logo_u8))
    signature = resize(sig_gray, (args.sig_size, args.sig_size), anti_aliasing=True)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[float, float, float, float, float]] = []
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "beta", "psnr_mean_db", "ncc_tw", "ncc_sig_biores"])
        for alpha in alphas:
            for beta in betas:
                psnr_m, tw_ncc, sig_ncc = _run_one(cover, biometric, signature, alpha, beta)
                w.writerow(
                    [
                        f"{alpha:g}",
                        f"{beta:g}",
                        f"{psnr_m:.4f}",
                        f"{tw_ncc:.4f}",
                        f"{sig_ncc:.4f}",
                    ]
                )
                rows.append((alpha, beta, psnr_m, tw_ncc, sig_ncc))

    # --- Best single objectives (higher is better for all three metrics here)
    best_psnr = max(rows, key=lambda r: r[2])
    best_tw = max(rows, key=lambda r: r[3])
    best_sig = max(rows, key=lambda r: r[4])

    # Balanced: normalize PSNR to ~[0,1] band and add weighted NCC(TW) (NCC can be negative)
    def balanced(r: tuple) -> float:
        _, _, ps, twc, _ = r
        return args.w_psnr * (ps / 70.0) + args.w_ncc_tw * twc

    best_bal = max(rows, key=balanced)

    def fmt(r: tuple) -> str:
        a, b, ps, nt, ns = r
        return f"α={a:g}, β={b:g} → PSNR={ps:.2f} dB, NCC_TW={nt:.6f}, NCC_sig={ns:.6f}"

    print(f"Wrote {args.csv} ({len(rows)} runs).")
    print()
    print("Best by each metric (from this grid):")
    print(f"  Highest PSNR (mean R,G,B):     {fmt(best_psnr)}")
    print(f"  Highest NCC (TW vs ext.):    {fmt(best_tw)}")
    print(f"  Highest NCC (sig vs ext.):     {fmt(best_sig)}")
    print()
    print(
        f"Best balanced score "
        f"({args.w_psnr}·PSNR/70 + {args.w_ncc_tw}·NCC_TW): {fmt(best_bal)}"
    )
    print()

    top_k = min(5, len(rows))
    by_psnr = sorted(rows, key=lambda r: -r[2])[:top_k]
    by_tw = sorted(rows, key=lambda r: -r[3])[:top_k]
    by_sig = sorted(rows, key=lambda r: -r[4])[:top_k]

    print(f"Top {top_k} by PSNR:")
    for r in by_psnr:
        print(" ", fmt(r))
    print(f"Top {top_k} by NCC(TW):")
    for r in by_tw:
        print(" ", fmt(r))
    print(f"Top {top_k} by NCC(signature):")
    for r in by_sig:
        print(" ", fmt(r))
    print()
    print(
        "Note: PSNR and NCC(TW) usually trade off (larger α,β → better NCC, worse PSNR). "
        "Pick the row that fits your thesis target, or change --alphas/--betas."
    )


if __name__ == "__main__":
    main()
