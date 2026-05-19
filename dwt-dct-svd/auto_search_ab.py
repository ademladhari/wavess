"""
Automatically search α and β (no manual α/β lists; default 100 evaluations).

Selection policy (after PSNR ≥ --psnr-floor filter, or fallback if none):
  1) Prefer NCC_TW ≥ --ncc-tw-target (default 0.9): watermark must be detectable.
  2) Among those, maximize mean PSNR (imperceptibility among good detectors).
  3) If none reach the NCC target, fall back to best NCC_TW then PSNR.

Uses an n_α × n_β grid from linspace (≈ square, e.g. 10×10 when --max-iter 100).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
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


def _load_data(
    root: Path,
    cover_size: int | None,
    bio_size: int,
    sig_size: int,
):
    orig_path = _find_image(root, "original")
    pic_path = _find_biometric_image(root)
    logo_path = _find_image(root, "logo")

    cover = _rgba_to_rgb_u8(io.imread(orig_path))
    if cover_size and cover_size > 0:
        s = int(cover_size)
        cover = resize(to_float01(cover), (s, s), anti_aliasing=True)
    picture_u8 = _rgba_to_rgb_u8(io.imread(pic_path))
    logo_u8 = _rgba_to_rgb_u8(io.imread(logo_path))

    bio_gray = color.rgb2gray(to_float01(picture_u8))
    biometric = resize(bio_gray, (bio_size, bio_size), anti_aliasing=True)
    sig_gray = color.rgb2gray(to_float01(logo_u8))
    signature = resize(sig_gray, (sig_size, sig_size), anti_aliasing=True)
    return cover, biometric, signature


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Auto 100-point (α,β) search: max NCC_TW with PSNR ≥ floor"
    )
    ap.add_argument(
        "--psnr-floor",
        type=float,
        default=50.0,
        help="Reject candidates with mean PSNR(R,G,B) below this (dB)",
    )
    ap.add_argument(
        "--cover-size",
        type=int,
        default=512,
        help="Cover resized to S×S; use 0 for no resize",
    )
    ap.add_argument("--bio-size", type=int, default=256)
    ap.add_argument("--sig-size", type=int, default=128)
    ap.add_argument(
        "--csv",
        type=Path,
        default=root / "output" / "auto_search_ab_results.csv",
    )
    ap.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Total evaluations (default 100 = 10×10 grid)",
    )
    ap.add_argument(
        "--ncc-tw-target",
        type=float,
        default=0.9,
        help="Secondary: NCC_TW threshold",
    )
    ap.add_argument(
        "--ncc-sig-target",
        type=float,
        default=0.95,
        help="Primary: prefer runs with NCC_signature ≥ this (watermark actually recoverable)",
    )
    args = ap.parse_args()

    n_total = max(1, int(args.max_iter))
    # ~square grid: n_a * n_b >= n_total; evaluate row-major and stop at n_total
    n_a = max(2, int(round(np.sqrt(n_total))))
    n_b = (n_total + n_a - 1) // n_a

    cover, biometric, signature = _load_data(
        root,
        args.cover_size if args.cover_size > 0 else None,
        args.bio_size,
        args.sig_size,
    )

    # Wide but reasonable ranges (no user input); tune here if needed.
    alpha_min, alpha_max = 0.015, 0.14
    beta_min, beta_max = 0.00025, 0.014

    alphas = np.linspace(alpha_min, alpha_max, n_a, dtype=np.float64)
    betas = np.linspace(beta_min, beta_max, n_b, dtype=np.float64)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[float, float, float, float, float]] = []
    iteration = 0

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["iteration", "alpha", "beta", "psnr_mean_db", "ncc_tw", "ncc_sig", "psnr_ok"]
        )
        for alpha in alphas:
            TW, state_alpha = embed_signature_in_biometric(
                biometric, signature, alpha=float(alpha)
            )
            for beta in betas:
                if iteration >= n_total:
                    break
                iteration += 1
                b = float(beta)
                a = float(alpha)
                watermarked, state = embed_tw_in_cover(
                    cover, TW, beta=b, state=state_alpha
                )
                # Realistic uint8 PNG round-trip so reported NCC matches the delivered image.
                wm_png = util.img_as_ubyte(np.clip(watermarked, 0.0, 1.0)).astype(
                    np.float64
                ) / 255.0
                psnr_m, _ = psnr_channelwise(cover, wm_png)
                TW_ext = extract_tw_from_watermarked(
                    wm_png, cover, beta=b, state=state
                )
                sig_ext = extract_signature_from_tw(
                    TW_ext, biometric, alpha=a, state=state
                )
                tw_ncc = ncc(TW, TW_ext)
                sig_ncc = ncc(signature, sig_ext)
                ok = psnr_m >= float(args.psnr_floor)
                w.writerow(
                    [
                        iteration,
                        f"{a:.6g}",
                        f"{b:.6g}",
                        f"{psnr_m:.6f}",
                        f"{tw_ncc:.6f}",
                        f"{sig_ncc:.6f}",
                        str(ok),
                    ]
                )
                rows.append((a, b, psnr_m, tw_ncc, sig_ncc))

            if iteration >= n_total:
                break

    feasible = [r for r in rows if r[2] >= float(args.psnr_floor)]
    tgt_sig = float(args.ncc_sig_target)
    tgt_tw = float(args.ncc_tw_target)

    def pick_best(pool: list[tuple[float, float, float, float, float]]) -> tuple:
        """Primary: maximize NCC_sig (the extracted-watermark correlation that Eq. (7)
        and Table 2/5 of the paper report). If NCC_sig ≥ tgt_sig, then prefer PSNR.
        Else best NCC_sig then NCC_tw then PSNR."""

        def key_good(r: tuple) -> tuple:
            _, _, ps, tw, sg = r
            ok = 1 if sg >= tgt_sig and tw >= tgt_tw else 0
            return (ok, sg, tw, ps) if ok else (0, sg, tw, ps)

        return max(pool, key=key_good)

    if feasible:
        best = pick_best(feasible)
        n_hit = sum(1 for r in feasible if r[4] >= tgt_sig and r[3] >= tgt_tw)
        status = (
            f"Best under PSNR ≥ {args.psnr_floor:g} dB: primary NCC_sig ≥ {tgt_sig:g} "
            f"(and NCC_TW ≥ {tgt_tw:g}): {n_hit}/{len(feasible)} hits, then highest PSNR:"
        )
    else:
        best = pick_best(rows)
        status = (
            f"WARNING: no pair reached PSNR ≥ {args.psnr_floor:g} dB in {n_total} tries. "
            f"Using same rule on all runs:"
        )

    print(f"Completed {len(rows)} evaluations (cap {n_total}). CSV: {args.csv}")
    print(status)
    print(
        f"  α={best[0]:.6g}, β={best[1]:.6g} → PSNR={best[2]:.2f} dB, "
        f"NCC_TW={best[3]:.6f}, NCC_sig={best[4]:.6f}"
    )
    if feasible:
        print(f"  Feasible count (PSNR ≥ {args.psnr_floor:g}): {len(feasible)} / {len(rows)}")
        n_hit = sum(1 for r in feasible if r[4] >= tgt_sig and r[3] >= tgt_tw)
        print(f"  NCC_sig ≥ {tgt_sig:g} AND NCC_TW ≥ {tgt_tw:g}: {n_hit} feasible runs")

        def key_good(r: tuple[float, float, float, float, float]) -> tuple:
            _, _, ps, tw, sg = r
            ok = 1 if sg >= tgt_sig and tw >= tgt_tw else 0
            return (ok, sg, tw, ps)

        feas_sorted = sorted(feasible, key=key_good, reverse=True)
        print("  Top 5 feasible (by NCC_sig, NCC_TW, PSNR):")
        for r in feas_sorted[:5]:
            print(
                f"    α={r[0]:.5g} β={r[1]:.5g}  PSNR={r[2]:.2f}  "
                f"NCC_TW={r[3]:.5f}  NCC_sig={r[4]:.5f}"
            )
    else:
        print("  Try lowering --psnr-floor or widening alpha/beta ranges in auto_search_ab.py")

    cs = int(args.cover_size) if args.cover_size > 0 else 0
    cs_arg = f" --cover-size {cs}" if cs > 0 else ""
    print(
        f"  Re-run pipeline with best pair: "
        f"python run_watermark.py{cs_arg} --alpha {best[0]:.8g} --beta {best[1]:.8g}"
    )


if __name__ == "__main__":
    main()
