"""Run simple benchmark for one or more watermark methods."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from wmbench.simple_benchmark.attacks import ATTACKS, AttackRunner
from wmbench.simple_benchmark.metrics import detection_auroc_and_tpr, psnr_ssim_pair
from wmbench.simple_benchmark.methods import (
    MethodProfile,
    compute_bit_accuracy,
    detection_score,
    negative_detection_score,
    payload_from_adapter,
    profile_for,
)
from wmbench.watermarks import get_adapter

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_adapter(method_id: str, device: torch.device):
    """Instantiate adapter; only pass device to constructors that accept it."""
    mid = method_id.strip().lower().replace("_", "-")
    kwargs: dict = {}
    if mid in {"flexible", "flex", "ssl", "robin", "dwt"}:
        kwargs["device"] = str(device)
    return get_adapter(method_id, **kwargs)


def _mean_finite(vals: list[float]) -> float:
    finite = [float(v) for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def list_images(root: Path, limit: int) -> list[Path]:
    paths = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
    if len(paths) < limit:
        raise FileNotFoundError(f"Need {limit} images under {root}, found {len(paths)}")
    return paths[:limit]


def center_crop_square_resize(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.LANCZOS)
    return im


def _embed_images(
    adapter,
    paths: list[Path],
    image_size: int,
    embed_batch_size: int,
) -> list[dict]:
    records: list[dict] = []
    batch_paths: list[Path] = []
    batch_originals: list[Image.Image] = []

    def flush_batch():
        nonlocal batch_paths, batch_originals
        if not batch_originals:
            return
        if embed_batch_size > 1 and hasattr(adapter, "embed_batch"):
            wms = adapter.embed_batch(batch_originals)
        else:
            wms = [adapter.embed(im) for im in batch_originals]
        for path, original, watermarked in zip(batch_paths, batch_originals, wms):
            meta = payload_from_adapter(adapter)
            records.append(
                {
                    "path": str(path),
                    "original": original,
                    "watermarked": watermarked,
                    "meta": meta,
                }
            )
        batch_paths = []
        batch_originals = []

    for path in tqdm(paths, desc="embed", unit="img"):
        with Image.open(path) as raw:
            original = center_crop_square_resize(raw, image_size)
        batch_paths.append(path)
        batch_originals.append(original)
        if len(batch_originals) >= embed_batch_size:
            flush_batch()
    flush_batch()
    return records


def _svd_payload_bank(records: list[dict]) -> list[dict]:
    bank = [r["meta"] for r in records if r.get("meta")]
    return bank


def benchmark_method(
    method_id: str,
    images_dir: Path,
    output_dir: Path,
    *,
    n_images: int = 1000,
    image_size: int = 512,
    device: torch.device,
    skip_regen: bool = False,
    blind_detect: bool | None = None,
    embed_batch_size: int = 1,
) -> list[dict]:
    profile = profile_for(method_id, blind_detect=blind_detect)
    method_out = output_dir / profile.method_id.replace("/", "_")
    method_out.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== method={profile.method_id} generation_based={profile.generation_based} "
        f"blind_detect={profile.blind_detect} device={device} ===",
        flush=True,
    )
    if profile.method_id == "dwt" and device.type == "cuda":
        print("dwt: GPU cross-correlation enabled (torch conv2d on CUDA)", flush=True)

    try:
        adapter = _load_adapter(method_id, device)
    except Exception as e:
        print(f"SKIP {method_id}: failed to load adapter ({e!r})", flush=True)
        err_path = method_out / "error.txt"
        err_path.write_text(str(e), encoding="utf-8")
        return []

    paths = list_images(images_dir, n_images)
    records = _embed_images(adapter, paths, image_size, max(1, embed_batch_size))
    if not records:
        raise RuntimeError(f"No embedded records for method {method_id}")

    shared_meta = records[0]["meta"] if getattr(adapter, "embed_meta_shared", False) else None
    svd_bank = _svd_payload_bank(records) if profile.method_id == "svd" else None

    neg_scores: list[float] = []
    for i, rec in enumerate(tqdm(records, desc="detect/negatives", unit="img")):
        neg_scores.append(
            negative_detection_score(
                adapter,
                rec["original"],
                rec["original"],
                profile,
                neg_meta=shared_meta if shared_meta is not None else rec.get("meta"),
                svd_payload_bank=svd_bank,
                neg_index=i,
            )
        )
    neg_arr = np.asarray(neg_scores, dtype=np.float64)

    runner = AttackRunner(device, skip_regen=skip_regen)
    rows: list[dict] = []

    for spec in ATTACKS:
        if spec.name == "regeneration" and skip_regen:
            continue

        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        bit_vals: list[float] = []
        pos_scores: list[float] = []
        track_bit_acc = profile.supports_bit_accuracy

        for rec in tqdm(records, desc=f"{profile.method_id}/{spec.name}", unit="img"):
            original: Image.Image = rec["original"]
            wm: Image.Image = rec["watermarked"]
            meta = rec["meta"] if not shared_meta else shared_meta

            attacked = runner.apply(spec.name, wm)
            p, s = psnr_ssim_pair(original, attacked)
            psnr_vals.append(p)
            ssim_vals.append(s)

            if track_bit_acc:
                bit_vals.append(
                    compute_bit_accuracy(adapter, original, attacked, meta, profile)
                )

            pos_scores.append(
                detection_score(
                    adapter,
                    attacked,
                    None if profile.blind_detect else original,
                    meta,
                    blind=profile.blind_detect,
                )
            )

        pos_arr = np.asarray(pos_scores, dtype=np.float64)
        auroc, tpr1 = detection_auroc_and_tpr(pos_arr, neg_arr, fpr_target=0.01)

        row = {
            "method": profile.method_id,
            "attack": spec.name,
            "description": spec.description,
            "n_images": len(records),
            "generation_based": profile.generation_based,
            "blind_detect": profile.blind_detect,
            "PSNR": float(np.mean(psnr_vals)),
            "SSIM": float(np.mean(ssim_vals)),
            "bit_accuracy": _mean_finite(bit_vals),
            "AUROC": auroc,
            "TPR_at_1pct_FPR": tpr1,
        }
        rows.append(row)
        bit_s = (
            f"{row['bit_accuracy']:.3f}"
            if np.isfinite(row["bit_accuracy"])
            else "n/a"
        )
        print(
            f"  {spec.name}: PSNR={row['PSNR']:.2f} SSIM={row['SSIM']:.3f} "
            f"BitAcc={bit_s} AUROC={row['AUROC']:.3f} "
            f"TPR@1%FPR={row['TPR_at_1pct_FPR']:.3f}",
            flush=True,
        )

    _write_method_csv(method_out / "results.csv", rows)
    with open(method_out / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": profile.method_id,
                "profile": profile.__dict__,
                "images_dir": str(images_dir),
                "n_images": len(records),
                "image_size": image_size,
                "attacks": rows,
            },
            f,
            indent=2,
        )
    return rows


def _write_method_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_benchmark(
    methods: list[str],
    images_dir: Path,
    output_dir: Path,
    **kwargs,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for method_id in methods:
        try:
            rows = benchmark_method(method_id, images_dir, output_dir, **kwargs)
            all_rows.extend(rows)
        except Exception as e:
            print(f"FAILED method={method_id}: {e!r}", flush=True)
            err_dir = output_dir / method_id.strip().lower().replace("_", "-")
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(repr(e), encoding="utf-8")

    if all_rows:
        _write_method_csv(output_dir / "results_all.csv", all_rows)
        with open(output_dir / "results_all.json", "w", encoding="utf-8") as f:
            json.dump({"methods": methods, "rows": all_rows}, f, indent=2)
        print(f"\nWrote {output_dir / 'results_all.csv'}", flush=True)

    return all_rows
