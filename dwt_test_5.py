#!/usr/bin/env python3
"""Quick DWT embed/detect test — per-image scores + AUROC on score vs peak_ratio."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn import metrics as sk_metrics

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from benchmark_attacks import apply_attack_rgb
from wmbench.watermarks import get_adapter
from wmbench.watermarks.dwt import _load_dwt_module

TEST_ATTACKS = ("identity", "rotation", "jpeg_q50", "blur", "combined")


def pick_images(n: int) -> list[Path]:
    for base in (_ROOT / "wmbench_data" / "images", _ROOT / "robinresults" / "images"):
        if not base.is_dir():
            continue
        paths = sorted(
            p
            for p in base.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and not p.name.startswith("wm_")
        )
        if len(paths) >= n:
            return paths[:n]
        clean = sorted(base.glob("clean_image_*.png"))
        if len(clean) >= n:
            return clean[:n]
    raise FileNotFoundError(f"Need at least {n} images under wmbench_data/images or robinresults/images")


def peak_ratio(original: Image.Image, attacked: Image.Image, meta: dict) -> float:
    impl = _load_dwt_module()
    o = np.asarray(original.convert("L"), dtype=float)
    c = np.asarray(attacked.convert("L"), dtype=float)
    if c.max() <= 1:
        c *= 255
    if o.max() <= 1:
        o *= 255
    _, _, recs = impl.detect_watermark_hierarchical(o, c, meta, ratio_threshold=1.05)
    return float(max(r["mean_peak_ratio"] for r in recs))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.zeros(neg.size, np.int32), np.ones(pos.size, np.int32)])
    s = np.concatenate([neg, pos])
    return float(sk_metrics.roc_auc_score(y, s))


def main() -> int:
    p = argparse.ArgumentParser(description="DWT smoke test")
    p.add_argument("-n", "--n-images", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    paths = pick_images(args.n_images)
    adapter = get_adapter("dwt", device="cpu")

    print(f"DWT test (n={args.n_images}, {args.size}x{args.size}, non-blind)", flush=True)
    print("Images:", ", ".join(p.name for p in paths))
    print()

    neg_scores: list[float] = []
    neg_peaks: list[float] = []

    for idx, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
        wm = adapter.embed(img)
        meta = adapter.payload_for_meta()
        assert meta is not None

        neg_scores.append(adapter.detect(img, img, meta=meta, blind=False))
        neg_peaks.append(peak_ratio(img, img, meta))

        print(f"--- {path.name} ---")
        for atk in TEST_ATTACKS:
            attacked = apply_attack_rgb(atk, wm, seed=idx)
            score = adapter.detect(attacked, img, meta=meta, blind=False)
            peak = peak_ratio(img, attacked, meta)
            print(f"  {atk:14s}  score={score:.3f}  peak_ratio={peak:.2f}")

        print(f"  {'clean (neg)':14s}  score={neg_scores[-1]:.3f}  peak_ratio={neg_peaks[-1]:.2f}")
        print()

    neg_score_arr = np.asarray(neg_scores, dtype=np.float64)
    neg_peak_arr = np.asarray(neg_peaks, dtype=np.float64)
    print(f"Neg mean: score={neg_score_arr.mean():.3f}  peak_ratio={neg_peak_arr.mean():.2f}")
    print()
    print(f"{'attack':<14} {'AUROC_score':>12} {'AUROC_peak':>12} {'mean_score':>11} {'mean_peak':>10}")
    print("-" * 62)

    for atk in TEST_ATTACKS:
        pos_scores: list[float] = []
        pos_peaks: list[float] = []
        for idx, path in enumerate(paths):
            img = Image.open(path).convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
            wm = adapter.embed(img)
            meta = adapter.payload_for_meta()
            assert meta is not None
            attacked = apply_attack_rgb(atk, wm, seed=idx)
            pos_scores.append(adapter.detect(attacked, img, meta=meta, blind=False))
            pos_peaks.append(peak_ratio(img, attacked, meta))

        pos_score_arr = np.asarray(pos_scores, dtype=np.float64)
        pos_peak_arr = np.asarray(pos_peaks, dtype=np.float64)
        auc_s = auroc(pos_score_arr, neg_score_arr)
        auc_p = auroc(pos_peak_arr, neg_peak_arr)
        print(
            f"{atk:<14} {auc_s:>12.3f} {auc_p:>12.3f} "
            f"{pos_score_arr.mean():>11.3f} {pos_peak_arr.mean():>10.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
