#!/usr/bin/env python3
"""
RingID / Tree-Ring benchmark for wavess + tree-ring-ringid (Colab-friendly).

Lives in the wavess repo so `git clone wavess` always has the runner.
Requires tree-ring-ringid cloned beside it (Colab cell 1 does this).

Example (Colab):
  python /content/wavess/ringid_benchmark.py --n-images 100 --output /content/out
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn import metrics as sk_metrics
from tqdm.auto import tqdm

WAVES_ROOT = Path(__file__).resolve().parent
RINGID_ROOT = WAVES_ROOT / "tree-ring-ringid"
SRC_ROOT = RINGID_ROOT / "src"

if str(WAVES_ROOT) not in sys.path:
    sys.path.insert(0, str(WAVES_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if not SRC_ROOT.is_dir():
    raise SystemExit(
        f"tree-ring-ringid not found at {RINGID_ROOT}\n"
        "Clone: git clone https://github.com/ademladhari/tree-ring-ringid "
        f"{RINGID_ROOT}"
    )

from benchmark_attacks import ATTACKS, apply_attack_rgb  # noqa: E402
from ringid.config import profile_ringid_default, profile_tree_ring_baseline  # noqa: E402
from ringid.detect import invert_then_patterns, verification_distances_vs_ref  # noqa: E402
from ringid.sampling import generate_clean_batch, generate_watermarked_batch, load_pipeline  # noqa: E402
from ringid.watermark import WatermarkKey  # noqa: E402


def load_prompts(path: Path, n: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No prompts in {path}")
    if len(lines) < n:
        reps = (n + len(lines) - 1) // len(lines)
        lines = (lines * reps)[:n]
    return lines[:n]


def build_profile(name: str, *, model_id: str | None, dtype: str | None, steps: int | None):
    profile = profile_tree_ring_baseline() if name == "tree_ring" else profile_ringid_default()
    if model_id:
        profile.model_id = model_id
    if dtype:
        profile.dtype = dtype
    if steps is not None:
        profile.num_inference_steps = int(steps)
    return profile


def psnr_ssim_pil(ref: Image.Image, attacked: Image.Image) -> tuple[float, float]:
    b = attacked.resize(ref.size, Image.Resampling.LANCZOS) if attacked.size != ref.size else attacked
    a_np = np.asarray(ref, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    psnr_v = float(peak_signal_noise_ratio(a_np, b_np, data_range=255.0))
    ssim_v = float(structural_similarity(a_np, b_np, channel_axis=2, data_range=255.0))
    return psnr_v, ssim_v


def distance_to_score(distance: float) -> float:
    return float(-distance)


def detection_auroc_and_tpr(
    pos: np.ndarray, neg: np.ndarray, fpr_target: float = 0.01
) -> tuple[float, float]:
    y_true = np.concatenate([np.zeros(neg.size, dtype=np.int32), np.ones(pos.size, dtype=np.int32)])
    y_score = np.concatenate([neg, pos])
    auroc = float(sk_metrics.roc_auc_score(y_true, y_score))
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score, pos_label=1)
    below = np.where(fpr < fpr_target)[0]
    tpr_at = float(tpr[below[-1]]) if below.size else float(tpr[0])
    return auroc, tpr_at


def run(
    output_dir: Path,
    *,
    n_images: int,
    prompts_path: Path,
    seed: int,
    watermark_seed: int,
    profile_name: str,
    model_id: str | None,
    dtype: str | None,
    num_steps: int | None,
    height: int,
    width: int,
    inversion_steps: int | None,
    gen_batch_size: int,
    detect_batch_size: int,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = build_profile(profile_name, model_id=model_id, dtype=dtype, steps=num_steps)
    prompts = load_prompts(prompts_path, n_images)
    gen_batch_size = max(1, int(gen_batch_size))
    detect_batch_size = max(1, int(detect_batch_size))

    pipe = load_pipeline(profile)
    inv_steps = inversion_steps if inversion_steps is not None else profile.num_inference_steps

    key_path = output_dir / "key.json"
    records: list[dict] = []

    print(
        f"Generating {n_images} watermarked + {n_images} clean images "
        f"(profile={profile_name}, model={profile.model_id}, steps={profile.num_inference_steps}, "
        f"gen_batch={gen_batch_size})…",
        flush=True,
    )
    ref_key: WatermarkKey | None = None
    for start in tqdm(range(0, n_images, gen_batch_size), desc="generate", unit="batch"):
        end = min(start + gen_batch_size, n_images)
        chunk_prompts = prompts[start:end]
        b = end - start
        latent_base = seed + start
        clean_seed_base = seed + 100_000 + start

        wm_images, _lat, key = generate_watermarked_batch(
            pipe,
            profile,
            prompt=chunk_prompts[0],
            prompts=chunk_prompts,
            negative_prompt="",
            height=height,
            width=width,
            n_images=b,
            latent_seed_base=latent_base,
            watermark_seed=watermark_seed,
        )
        if ref_key is None:
            ref_key = key
            ref_key.save_json(key_path)
        elif not torch.allclose(ref_key.vector, key.vector, atol=1e-4, rtol=1e-5):
            raise RuntimeError("Watermark key drifted across images — check watermark_seed.")

        clean_images = generate_clean_batch(
            pipe,
            profile,
            chunk_prompts,
            negative_prompt="",
            height=height,
            width=width,
            seed_base=clean_seed_base,
        )

        for j in range(b):
            i = start + j
            records.append(
                {
                    "prompt": chunk_prompts[j],
                    "wm_pil": wm_images[j],
                    "clean_pil": clean_images[j],
                    "latent_seed": seed + i,
                    "clean_seed": seed + 100_000 + i,
                }
            )

    assert ref_key is not None and key_path.is_file()

    def detect_distances_batch(pil_images: list[Image.Image], prompt_list: list[str]) -> list[float]:
        out_d: list[float] = []
        for s in range(0, len(pil_images), detect_batch_size):
            chunk_imgs = pil_images[s : s + detect_batch_size]
            chunk_prompts = prompt_list[s : s + detect_batch_size]
            patterns = invert_then_patterns(
                pipe,
                pil_images=chunk_imgs,
                prompts=chunk_prompts,
                negative_prompt="",
                profile=profile,
                num_inference_steps=inv_steps,
            )
            for pat in patterns:
                scores = verification_distances_vs_ref(pat, genuine_key_json=key_path)
                out_d.append(float(scores["d_wm_to_w"]))
        return out_d

    sanity_d = detect_distances_batch([records[0]["wm_pil"]], [records[0]["prompt"]])[0]
    print(f"Sanity (image 0, no attack): d_wm_to_w={sanity_d:.1f} score={distance_to_score(sanity_d):.1f}", flush=True)

    neg_dists = detect_distances_batch(
        [rec["clean_pil"] for rec in records],
        [rec["prompt"] for rec in records],
    )
    neg_scores = np.asarray([distance_to_score(d) for d in neg_dists], dtype=np.float64)

    rows_out: list[dict] = []
    for spec in ATTACKS:
        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        attacked_all: list[Image.Image] = []
        prompt_all: list[str] = []

        for i, rec in enumerate(records):
            attacked_pil = apply_attack_rgb(spec.name, rec["wm_pil"], seed=i)
            p, s = psnr_ssim_pil(rec["wm_pil"], attacked_pil)
            psnr_vals.append(p)
            ssim_vals.append(s)
            attacked_all.append(attacked_pil)
            prompt_all.append(rec["prompt"])

        pos_dists = detect_distances_batch(attacked_all, prompt_all)
        pos_scores = [distance_to_score(d) for d in pos_dists]

        pos_arr = np.asarray(pos_scores, dtype=np.float64)
        auroc, tpr1 = detection_auroc_and_tpr(pos_arr, neg_scores)

        row = {
            "method": "ringid" if profile_name == "ring_id" else "tree_ring_ringid",
            "detector": "ddim_invert_l1",
            "profile": profile_name,
            "attack": spec.name,
            "description": spec.description,
            "n_images": len(records),
            "PSNR": float(np.mean(psnr_vals)),
            "SSIM": float(np.mean(ssim_vals)),
            "bit_accuracy": float("nan"),
            "AUROC": auroc,
            "TPR_at_1pct_FPR": tpr1,
        }
        rows_out.append(row)
        print(
            f"  {spec.name}: PSNR={row['PSNR']:.2f} SSIM={row['SSIM']:.3f} "
            f"AUROC={row['AUROC']:.3f} TPR@1%FPR={row['TPR_at_1pct_FPR']:.3f}",
            flush=True,
        )

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    summary = {
        "method": "ringid" if profile_name == "ring_id" else "tree_ring_ringid",
        "implementation": "tree-ring-ringid/src/ringid",
        "generation_based": True,
        "profile": profile_name,
        "model_id": profile.model_id,
        "n_images": len(records),
        "seed": seed,
        "watermark_seed": watermark_seed,
        "gen_batch_size": gen_batch_size,
        "detect_batch_size": detect_batch_size,
        "prompts_file": str(prompts_path),
        "key_json": str(key_path),
        "sanity_d_wm_to_w": sanity_d,
        "attacks": rows_out,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {csv_path}", flush=True)
    return rows_out


def main() -> int:
    p = argparse.ArgumentParser(description="RingID / Tree-Ring benchmark (WAVES attacks)")
    p.add_argument("--output", type=Path, default=RINGID_ROOT / "outputs_benchmark")
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--prompts", type=Path, default=WAVES_ROOT / "tree-ring" / "prompts.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--watermark-seed", type=int, default=42)
    p.add_argument("--profile", choices=("ring_id", "tree_ring"), default="ring_id")
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--dtype", choices=("float16", "float32"), default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--inversion-steps", type=int, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--gen-batch-size", type=int, default=4)
    p.add_argument("--detect-batch-size", type=int, default=2)
    args = p.parse_args()

    run(
        args.output,
        n_images=args.n_images,
        prompts_path=args.prompts,
        seed=args.seed,
        watermark_seed=args.watermark_seed,
        profile_name=args.profile,
        model_id=args.model_id,
        dtype=args.dtype,
        num_steps=args.steps,
        height=args.height,
        width=args.width,
        inversion_steps=args.inversion_steps,
        gen_batch_size=args.gen_batch_size,
        detect_batch_size=args.detect_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
