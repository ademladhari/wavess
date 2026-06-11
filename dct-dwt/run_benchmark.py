#!/usr/bin/env python3
"""
Standalone DCT-DWT benchmark using dct-dwt/dwt_dct_watermarking.ipynb.

Blind PN-sequence embedding in 2-level DWT + 4x4 DCT mid-band (Al-Haj style).

Metrics per attack:
  - PSNR, SSIM (host grayscale vs watermarked+attacked)
  - Bit accuracy (extracted bits vs embedded payload)
  - AUROC, TPR @ 1% FPR (score = (rho + 1) / 2, blind key-based detector)

Attacks: see benchmark_attacks.py (rotation, blur, brightness, JPEG, crop, etc.)

Default embed gain α is the paper/notebook value (~0.073, host PSNR ~95 dB).
Override with --alpha 10 for a stronger (more visible) embed.

Example:
  python run_benchmark.py --images "D:\\new method\\data\\coco100k\\val" --n-images 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn import metrics
from tqdm.auto import tqdm

_WAVES_ROOT = Path(__file__).resolve().parents[1]
if str(_WAVES_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAVES_ROOT))

from benchmark_attacks import ATTACKS, apply_attack_gray  # noqa: E402
from load_impl import load_notebook_impl

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path, limit: int) -> list[Path]:
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if len(paths) < limit:
        raise FileNotFoundError(f"Need {limit} images under {root}, found {len(paths)}")
    return paths[:limit]


def load_gray(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(im.convert("L"), dtype=np.float64)


def make_seeded_wm_bits(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = (rng.integers(0, 2, size=(size, size), dtype=np.uint8) * 255).astype(np.uint8)
    return (arr.flatten() > 127).astype(np.uint8)


def load_wm_bits(
    impl,
    watermark_path: Path | None,
    wm_seed: int,
    embed_wm_size: int,
    *,
    use_qr_default: bool,
) -> np.ndarray:
    if watermark_path is not None and watermark_path.is_file():
        wm = impl.load_grayscale(str(watermark_path), embed_wm_size)
        return impl.image_to_bits(wm)
    if use_qr_default:
        qr = Path(__file__).resolve().parent / "pictures" / "qrcodetest1.png"
        if qr.is_file():
            wm = impl.load_grayscale(str(qr), embed_wm_size)
            return impl.image_to_bits(wm)
    return make_seeded_wm_bits(wm_seed, embed_wm_size)


def decoder_sanity(
    impl,
    host: np.ndarray,
    watermarked: np.ndarray,
    payload: dict,
) -> dict:
    """Detect collapsed decoder (constant output regardless of embed)."""
    ex_h = extract_bits(impl, host, payload)
    ex_w = extract_bits(impl, watermarked, payload)
    ref = np.asarray(payload["wm_bits"], dtype=np.uint8).flatten()
    ones_frac = float(ref.mean())
    collapsed = bool(np.array_equal(ex_h, ex_w) and len(np.unique(ex_w)) <= 1)
    spurious_acc = bool(
        collapsed and abs(float(np.mean(ref[: ex_w.size] == ex_w)) - ones_frac) < 0.02
    )
    return {
        "decoder_collapsed": collapsed,
        "spurious_bit_accuracy": spurious_acc,
        "payload_ones_frac": ones_frac,
        "extracted_unique_bits": int(len(np.unique(ex_w))),
        "identity_bit_accuracy": float(bit_accuracy(ref, ex_w)),
        "clean_bit_accuracy": float(bit_accuracy(ref, ex_h)),
    }


def extract_bits(impl, attacked: np.ndarray, payload: dict) -> np.ndarray:
    return impl.extract_watermark_bits(
        attacked,
        capacity=int(payload["capacity"]),
        seed=int(payload["seed"]),
        wavelet=str(payload["wavelet"]),
        subband_choice=str(payload["subband_choice"]),
        mid_pos=[tuple(x) for x in payload["mid_pos"]],
        pn_mode=str(payload["pn_mode"]),
    )


def bit_accuracy(ref_bits: np.ndarray, ex_bits: np.ndarray) -> float:
    n = min(ref_bits.size, ex_bits.size)
    return float(np.mean(ref_bits[:n] == ex_bits[:n].astype(np.uint8)))


def detect_score(impl, attacked: np.ndarray, payload: dict) -> float:
    ex = extract_bits(impl, attacked, payload)
    ref = np.asarray(payload["wm_bits"], dtype=np.uint8).flatten()[: ex.size]
    rho = float(impl.rho(ref, ex.astype(np.uint8)))
    return float(np.clip((rho + 1.0) * 0.5, 0.0, 1.0))


def detection_auroc_and_tpr(
    pos: np.ndarray, neg: np.ndarray, fpr_target: float = 0.01
) -> tuple[float, float]:
    y_true = np.concatenate(
        [np.zeros(neg.size, dtype=np.int32), np.ones(pos.size, dtype=np.int32)]
    )
    y_score = np.concatenate([neg, pos])
    auroc = float(metrics.roc_auc_score(y_true, y_score))
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    below = np.where(fpr < fpr_target)[0]
    tpr_at = float(tpr[below[-1]]) if below.size else float(tpr[0])
    return auroc, tpr_at


def run(
    images_dir: Path,
    output_dir: Path,
    *,
    n_images: int,
    image_size: int,
    alpha: float | None,
    seed: int | None,
    wm_seed: int,
    watermark_path: Path | None,
    use_qr_default: bool,
    pn_mode: str | None,
    notebook: Path | None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    impl = load_notebook_impl(notebook)

    host_size = int(impl.HOST_SIZE)
    if image_size != host_size:
        print(f"Note: notebook HOST_SIZE={host_size}; using {host_size} for embed/extract.")
        image_size = host_size

    alpha_v = float(alpha if alpha is not None else impl.ALPHA)  # paper-style default ~0.073
    seed_v = int(seed if seed is not None else impl.SEED)
    wavelet = str(impl.WAVELET)
    subband = str(impl.SUBBAND_CHOICE)
    pn_mode = str(pn_mode if pn_mode is not None else "orthogonal_normal")
    if pn_mode == "independent_rademacher":
        print(
            "WARNING: pn_mode=independent_rademacher often outputs constant bits on "
            "natural images (bit accuracy becomes payload ones fraction). "
            "Prefer --pn-mode orthogonal_normal.",
            flush=True,
        )
    mid_pos = [tuple(x) for x in impl.MID_POS]
    embed_wm_size = int(impl.EMBED_WM_SIZE)

    wm_bits = load_wm_bits(
        impl, watermark_path, wm_seed, embed_wm_size, use_qr_default=use_qr_default
    )
    paths = list_images(images_dir, n_images)

    if float(wm_bits.mean()) > 0.9 or float(wm_bits.mean()) < 0.1:
        print(
            f"WARNING: payload is heavily skewed ({wm_bits.mean():.1%} ones). "
            "Bit accuracy can look high when the decoder outputs a constant. "
            "Prefer balanced payload (default) or --wm-seed.",
            flush=True,
        )

    records: list[dict] = []
    print(
        f"Embedding {len(paths)} images "
        f"(host={image_size}, wm_bits={embed_wm_size}x{embed_wm_size}, alpha={alpha_v}, seed={seed_v})…",
        flush=True,
    )
    for path in tqdm(paths, desc="embed", unit="img"):
        host = load_gray(path, image_size)
        wm_img, state = impl.embed_watermark(
            host,
            wm_bits,
            alpha=alpha_v,
            seed=seed_v,
            wavelet=wavelet,
            subband_choice=subband,
            mid_pos=mid_pos,
            pn_mode=pn_mode,
        )
        payload = {
            "capacity": int(state["capacity"]),
            "seed": seed_v,
            "wavelet": wavelet,
            "subband_choice": subband,
            "mid_pos": mid_pos,
            "pn_mode": pn_mode,
            "wm_bits": wm_bits,
        }
        records.append(
            {
                "path": str(path),
                "host": host,
                "watermarked": wm_img,
                "payload": payload,
            }
        )

    sanity = decoder_sanity(impl, records[0]["host"], records[0]["watermarked"], records[0]["payload"])
    if sanity["decoder_collapsed"]:
        print(
            "ERROR: decoder outputs identical constant bits on clean vs watermarked host. "
            f"pn_mode={pn_mode!r} alpha={alpha_v}. Bit accuracy and AUROC are not meaningful.",
            flush=True,
        )
    elif sanity["spurious_bit_accuracy"]:
        print(
            "WARNING: bit accuracy may equal payload ones fraction (decoder nearly constant).",
            flush=True,
        )
    print(
        f"Sanity (image 0): clean BitAcc={sanity['clean_bit_accuracy']:.3f} "
        f"wm BitAcc={sanity['identity_bit_accuracy']:.3f} "
        f"extracted_unique={sanity['extracted_unique_bits']} "
        f"payload_ones={sanity['payload_ones_frac']:.3f}",
        flush=True,
    )

    neg_scores = np.asarray(
        [
            detect_score(impl, rec["host"], rec["payload"])
            for rec in tqdm(records, desc="detect/negatives", unit="img")
        ],
        dtype=np.float64,
    )

    rows_out: list[dict] = []
    for spec in ATTACKS:
        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        bit_vals: list[float] = []
        pos_scores: list[float] = []

        for i, rec in enumerate(tqdm(records, desc=f"dct-dwt/{spec.name}", unit="img")):
            host = rec["host"]
            attacked = apply_attack_gray(spec.name, rec["watermarked"], seed=i)
            psnr_vals.append(float(impl.psnr(host, attacked)))
            ssim_vals.append(float(impl.ssim(host, attacked)))
            ex = extract_bits(impl, attacked, rec["payload"])
            bit_vals.append(bit_accuracy(rec["payload"]["wm_bits"], ex))
            pos_scores.append(detect_score(impl, attacked, rec["payload"]))

        pos_arr = np.asarray(pos_scores, dtype=np.float64)
        auroc, tpr1 = detection_auroc_and_tpr(pos_arr, neg_scores)

        row = {
            "method": "dct-dwt",
            "detector": "blind_pn_rho",
            "attack": spec.name,
            "description": spec.description,
            "n_images": len(records),
            "PSNR": float(np.mean(psnr_vals)),
            "SSIM": float(np.mean(ssim_vals)),
            "bit_accuracy": float(np.mean(bit_vals)),
            "AUROC": auroc,
            "TPR_at_1pct_FPR": tpr1,
        }
        rows_out.append(row)
        print(
            f"  {spec.name}: PSNR={row['PSNR']:.2f} SSIM={row['SSIM']:.3f} "
            f"BitAcc={row['bit_accuracy']:.3f} AUROC={row['AUROC']:.3f} "
            f"TPR@1%FPR={row['TPR_at_1pct_FPR']:.3f}",
            flush=True,
        )

    csv_path = output_dir / "results.csv"
    fields = list(rows_out[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "method": "dct-dwt",
        "implementation": "dct-dwt/dwt_dct_watermarking.ipynb",
        "detector": "blind extract_watermark_bits + rho",
        "images_dir": str(images_dir),
        "n_images": len(records),
        "image_size": image_size,
        "embed_wm_size": embed_wm_size,
        "alpha": alpha_v,
        "seed": seed_v,
        "wm_seed": wm_seed,
        "wavelet": wavelet,
        "subband_choice": subband,
        "pn_mode": pn_mode,
        "decoder_sanity": sanity,
        "attacks": rows_out,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {csv_path}", flush=True)
    return rows_out


def main() -> int:
    p = argparse.ArgumentParser(description="DCT-DWT standalone benchmark")
    p.add_argument(
        "--images",
        type=Path,
        default=Path(r"D:\new method\data\coco100k\val"),
    )
    p.add_argument("--output", type=Path, default=Path("outputs_benchmark"))
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="embed gain α (default: notebook/paper ~0.073, PSNR ~95 dB; use e.g. 10 for stronger embed)",
    )
    p.add_argument("--seed", type=int, default=None, help="PN sequence seed")
    p.add_argument("--wm-seed", type=int, default=777, help="balanced random payload seed")
    p.add_argument(
        "--watermark",
        type=Path,
        default=None,
        help="payload image (optional; default is balanced random bits)",
    )
    p.add_argument(
        "--watermark-qr",
        action="store_true",
        help="use pictures/qrcodetest1.png payload (98%% ones — skews bit accuracy)",
    )
    p.add_argument(
        "--pn-mode",
        type=str,
        default="orthogonal_normal",
        choices=["orthogonal_normal", "independent_rademacher", "independent_normal", "antipodal"],
        help="PN pair mode (notebook independent_rademacher collapses on natural images)",
    )
    p.add_argument("--notebook", type=Path, default=None)
    args = p.parse_args()

    run(
        args.images,
        args.output,
        n_images=args.n_images,
        image_size=args.image_size,
        alpha=args.alpha,
        seed=args.seed,
        wm_seed=args.wm_seed,
        watermark_path=args.watermark,
        use_qr_default=args.watermark_qr,
        pn_mode=args.pn_mode,
        notebook=args.notebook,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
