#!/usr/bin/env python3
"""DCT-DWT benchmark — blind bit decode (BER) on WAVES 14-attack suite."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn import metrics as sk_metrics

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from benchmark_attacks import ATTACKS, apply_attack_rgb
from wmbench.watermarks import get_adapter

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ATTACK_LABELS = {
    "identity": "Identity",
    "rotation": "Rotation 22.5°",
    "resized_crop": "Resized crop ×0.75",
    "erasing": "Random erasing 12.5%",
    "brightness": "Brightness ×1.5",
    "contrast": "Contrast ×1.5",
    "blur": "Gaussian blur r=4",
    "resize_90": "Resize 90% → back",
    "jpeg_q50": "JPEG Q50",
    "crop": "Crop 85% → back",
    "gaussian": "Gaussian noise σ=25",
    "combo_geometric": "Combo: geo",
    "combo_photometric": "Combo: photo",
    "combined": "Combined",
}


def list_images(root: Path, n: int) -> list[Path]:
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if len(paths) < n:
        raise FileNotFoundError(f"Need {n} images under {root}, found {len(paths)}")
    return paths[:n]


def pick_images(n: int) -> list[Path]:
    for base in (_ROOT / "wmbench_data" / "images", _ROOT / "robinresults" / "images"):
        if not base.is_dir():
            continue
        try:
            return list_images(base, n)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Need at least {n} images under wmbench_data/images or robinresults/images"
    )


def load_rgb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        return im


def bit_accuracy(adapter, attacked: Image.Image, meta: dict) -> float:
    impl = adapter._impl  # type: ignore[attr-defined]
    host = adapter._load_host_from_image(attacked, f"ba_{id(attacked)}")  # type: ignore[attr-defined]
    ex_bits = impl.extract_watermark_bits(
        host,
        capacity=int(meta["capacity"]),
        seed=int(meta["seed"]),
        wavelet=str(meta["wavelet"]),
        subband_choice=str(meta["subband_choice"]),
        mid_pos=[tuple(x) for x in meta["mid_pos"]],
        pn_mode=str(meta["pn_mode"]),
    )
    ref = np.asarray(meta["wm_bits"], dtype=np.uint8).flatten()[: len(ex_bits)]
    return float(np.mean(ref == ex_bits.astype(np.uint8)))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.zeros(neg.size, np.int32), np.ones(pos.size, np.int32)])
    s = np.concatenate([neg, pos])
    return float(sk_metrics.roc_auc_score(y, s))


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr_target: float = 0.01) -> float:
    y = np.concatenate([np.zeros(neg.size, np.int32), np.ones(pos.size, np.int32)])
    s = np.concatenate([neg, pos])
    fpr, tpr, _ = sk_metrics.roc_curve(y, s, pos_label=1)
    below = np.where(fpr < fpr_target)[0]
    return float(tpr[below[-1]]) if below.size else float(tpr[0])


def main() -> int:
    p = argparse.ArgumentParser(description="DCT-DWT benchmark (14 WAVES attacks, BER)")
    p.add_argument(
        "--images",
        type=Path,
        default=None,
        help="image folder (default: wmbench_data/images; thesis set: D:\\new method\\data\\coco100k\\val)",
    )
    p.add_argument("-n", "--n-images", type=int, default=5)
    p.add_argument("--size", type=int, default=512, help="HOST_SIZE (default 512)")
    p.add_argument("--output", type=Path, default=None, help="write results.csv here")
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="embed gain (default: notebook ~0.073)",
    )
    p.add_argument(
        "--pn-mode",
        default="orthogonal_normal",
        help="PN mode (default orthogonal_normal)",
    )
    args = p.parse_args()

    if args.images is not None:
        paths = list_images(args.images, args.n_images)
        image_dir = args.images
    else:
        paths = pick_images(args.n_images)
        image_dir = paths[0].parent

    adapter_kw: dict = {"pn_mode": args.pn_mode}
    if args.alpha is not None:
        adapter_kw["alpha"] = args.alpha
    adapter = get_adapter("dct-dwt", **adapter_kw)

    alpha_s = getattr(adapter, "_alpha", "?")
    pn_s = getattr(adapter, "_pn_mode", "?")
    print(
        f"DCT-DWT benchmark (n={args.n_images}, blind BER, alpha={alpha_s}, pn_mode={pn_s})",
        flush=True,
    )
    print(f"Images: {image_dir}", flush=True)
    print()

    records: list[dict] = []
    for idx, path in enumerate(paths):
        img = load_rgb(path, args.size)
        wm = adapter.embed(img)
        meta = adapter.payload_for_meta()
        assert meta is not None
        records.append(
            {
                "path": path,
                "clean": img,
                "wm": wm,
                "meta": meta,
                "neg_score": adapter.detect(img, meta=meta, blind=True),
                "neg_bits": bit_accuracy(adapter, img, meta),
            }
        )

    neg_scores = np.asarray([r["neg_score"] for r in records], dtype=np.float64)
    neg_bits = np.asarray([r["neg_bits"] for r in records], dtype=np.float64)
    print(
        f"Clean mean: bit_acc={neg_bits.mean():.3f}  BER={100*(1-neg_bits.mean()):.1f}%  "
        f"det_score={neg_scores.mean():.3f}",
        flush=True,
    )
    print()

    rows_out: list[dict] = []
    print(f"{'attack':<22} {'BER%':>7} {'bit_acc':>8} {'AUROC':>8} {'TPR@1%':>8}")
    print("-" * 58)

    for spec in ATTACKS:
        pos_scores: list[float] = []
        pos_bits: list[float] = []
        for i, rec in enumerate(records):
            attacked = apply_attack_rgb(spec.name, rec["wm"], seed=i)
            pos_scores.append(adapter.detect(attacked, meta=rec["meta"], blind=True))
            pos_bits.append(bit_accuracy(adapter, attacked, rec["meta"]))

        pos_score_arr = np.asarray(pos_scores, dtype=np.float64)
        pos_bit_arr = np.asarray(pos_bits, dtype=np.float64)
        mean_bit = float(pos_bit_arr.mean())
        mean_ber = 100.0 * (1.0 - mean_bit)
        auc = auroc(pos_score_arr, neg_scores)
        tpr1 = tpr_at_fpr(pos_score_arr, neg_scores)

        row = {
            "method": "dct-dwt",
            "detector": "blind_pn_rho",
            "attack": spec.name,
            "description": spec.description,
            "n_images": len(records),
            "bit_accuracy": mean_bit,
            "BER_pct": mean_ber,
            "AUROC": auc,
            "TPR_at_1pct_FPR": tpr1,
        }
        rows_out.append(row)
        label = ATTACK_LABELS.get(spec.name, spec.name)
        print(f"{label:<22} {mean_ber:7.1f} {mean_bit:8.3f} {auc:8.3f} {tpr1:8.3f}")

    print()
    print("Thesis BER column (DCT-DWT):")
    for spec in ATTACKS:
        ber = next(r["BER_pct"] for r in rows_out if r["attack"] == spec.name)
        label = ATTACK_LABELS.get(spec.name, spec.name)
        print(f"  {label:<22} {ber:.1f}")

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        csv_path = args.output / "results.csv"
        fields = list(rows_out[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWrote {csv_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
