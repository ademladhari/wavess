from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from wmbench.metrics import aggregate as agg
from wmbench.metrics.distribution import (
    compute_clip_fid,
    compute_clip_fid_from_dirs,
    compute_fid_metric,
    compute_fid_from_dirs,
)
from wmbench.metrics.image_similarity import compute_nmi, compute_psnr, compute_ssim
from wmbench.metrics.aesthetics import compute_aesthetics_and_artifacts_scores
from wmbench.metrics.quality import compute_aesthetics_delta_and_artifacts, compute_lpips
from wmbench.pipeline.detect import tpr_at_fpr
from wmbench.pipeline.resume import is_done, mark_done


def _watermarked_image_basenames(watermarked_dir: str) -> list[str]:
    names: list[str] = []
    for p in sorted(glob.glob(os.path.join(watermarked_dir, "*"))):
        if not os.path.isfile(p) or ".wmbench_meta" in os.path.basename(p):
            continue
        if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            names.append(os.path.basename(p))
    return names


def _ensure_fid_reference_mirror(work_dir: str, originals_dir: str) -> str | None:
    """Copy paired originals into a stable folder so clean-fid caches stats once (not per tempdir)."""
    wm = os.path.join(work_dir, "watermarked")
    basenames = _watermarked_image_basenames(wm)
    if not basenames:
        return None
    mirror = os.path.join(work_dir, "_wmbench_metric_cache", "fid_reference_images")
    manifest = os.path.join(mirror, ".wmbench_manifest.txt")
    key = hashlib.sha1(
        (os.path.normcase(os.path.abspath(originals_dir)) + "\n" + "\n".join(basenames)).encode(),
        usedforsecurity=False,
    ).hexdigest()
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as mf:
            if mf.read().strip() == key:
                return mirror
    if os.path.isdir(mirror):
        shutil.rmtree(mirror)
    os.makedirs(mirror, exist_ok=True)
    for b in basenames:
        src = os.path.join(originals_dir, b)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(mirror, b))
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write(key)
    return mirror


def _load_pairs(attacked_dir: str, originals_dir: str) -> tuple[list[Image.Image], list[Image.Image], list[str]]:
    paths = sorted(
        p
        for p in glob.glob(os.path.join(attacked_dir, "*"))
        if os.path.isfile(p) and p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    )
    originals: list[Image.Image] = []
    attacked: list[Image.Image] = []
    for ap in paths:
        base = os.path.basename(ap)
        op = os.path.join(originals_dir, base)
        if not os.path.isfile(op):
            continue
        with Image.open(op) as oim:
            originals.append(oim.convert("RGB"))
        with Image.open(ap) as aim:
            attacked.append(aim.convert("RGB"))
    basenames = [os.path.basename(p) for p in paths if os.path.isfile(os.path.join(originals_dir, os.path.basename(p)))]
    return originals, attacked, basenames


def run_evaluate_stage(
    work_dir: str,
    originals_dir: str,
    attack_names: list[str],
    strength_values: dict[str, list],
    *,
    output_dir: str,
    resume: bool = False,
    device: torch.device | None = None,
    skip_aesthetics_metrics: bool = False,
    lpips_batch_size: int = 1,
    profile_metrics: bool = False,
) -> None:
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    aesthetics_dev = dev
    if not skip_aesthetics_metrics:
        forced_aesth_dev = os.environ.get("WMBENCH_AESTHETICS_DEVICE", "").strip().lower()
        if forced_aesth_dev:
            aesthetics_dev = torch.device(forced_aesth_dev)
        elif os.name == "nt" and dev.type == "cuda":
            # Keep metrics enabled on Windows while avoiding known native CLIP CUDA load crashes.
            aesthetics_dev = torch.device("cpu")
            print(
                "evaluate: Windows+CUDA detected; loading aesthetics models on CPU for stability "
                "(override with WMBENCH_AESTHETICS_DEVICE=cuda).",
                flush=True,
            )
    attacked_root = os.path.join(work_dir, "attacked")
    scores_root = os.path.join(work_dir, "scores")
    metrics_root = os.path.join(work_dir, "metrics")

    aesthetics_logged = False
    aesthetics_orig_rating_cache: dict[tuple[str, ...], list[float]] = {}
    metric_progress = True
    lpips_batch_size = max(1, int(lpips_batch_size))
    stage_t0 = time.perf_counter()
    if profile_metrics and dev.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            pass

    def _require_finite(name: str, value: float, attack_name: str, strength_tag: str) -> float:
        if np.isnan(value) or np.isinf(value):
            raise RuntimeError(
                f"{name} produced non-finite value for attack={attack_name}, strength={strength_tag}: {value}"
            )
        return float(value)

    eval_cells: list[tuple[str, float | int]] = []
    for attack_name in attack_names:
        for strength in strength_values.get(attack_name, []):
            eval_cells.append((attack_name, strength))

    # Pending jobs: skip resume-done here; missing scores get mark_done only (same as before).
    jobs: dict[tuple[str, str], dict[str, Any]] = {}
    for attack_name, strength in eval_cells:
        stren_tag = str(strength).replace(os.sep, "_")
        attacked_dir = os.path.join(attacked_root, attack_name, stren_tag)
        out_dir = os.path.join(metrics_root, attack_name, stren_tag)
        done_flag = os.path.join(out_dir, ".done")
        if resume and is_done(done_flag):
            continue
        os.makedirs(out_dir, exist_ok=True)

        scores_path = os.path.join(scores_root, attack_name, stren_tag, "scores.json")
        if not os.path.isfile(scores_path):
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "missing_components.txt"), "a", encoding="utf-8") as mf:
                mf.write(f"scores missing skip metrics: {scores_path}\n")
            mark_done(done_flag)
            continue

        with open(scores_path, encoding="utf-8") as sf:
            sc = json.load(sf)
        pos = np.array(sc.get("positive") or [], dtype=float)
        neg = np.array(sc.get("negative") or [], dtype=float)
        try:
            p_det = tpr_at_fpr(pos, neg, fpr_target=0.001)
        except Exception:
            p_det = float("nan")

        key = (attack_name, stren_tag)
        jobs[key] = {
            "attack_name": attack_name,
            "stren_tag": stren_tag,
            "attacked_dir": attacked_dir,
            "out_dir": out_dir,
            "done_flag": done_flag,
            "p_det": float(p_det),
            "row": None,  # filled in phases
        }

    job_keys = sorted(jobs.keys(), key=lambda k: (k[0], str(k[1])))

    n_pending = len(job_keys)
    n_total_cells = len(eval_cells)
    if n_pending == 0:
        print(
            f"evaluate: 0 pending metric cells ({n_total_cells} total); "
            "all already have metrics/.done (use fresh output dir or delete work/.../metrics/**/.done to re-run). "
            "Skipping LPIPS / FID / CLIP-FID / aesthetics model load.",
            flush=True,
        )
        return

    print(
        f"evaluate: {n_pending} pending metric cell(s) (of {n_total_cells}); loading models…\n"
        f"evaluate: pipeline_file={os.path.abspath(__file__)}",
        flush=True,
    )

    # Load heavy metric models only when there is work (avoid minute-long load on resume-all-done).
    # These loads can take minutes (CLIP ViT-H, HF hub I/O) with no tqdm — log each step so it never looks hung.
    # If the process vanishes here with NO Python traceback and NO evaluate_crash.txt, suspect a native
    # CUDA/driver crash or OS kill (OOM); try --device cpu once to confirm.
    if dev.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    lpips_model = None
    try:
        print("evaluate: loading LPIPS (alex)…", flush=True)
        t_lm = time.perf_counter()
        from wmbench.metrics.perceptual import load_perceptual_models

        lpips_model = load_perceptual_models("lpips", "alex", device=dev)
        print(f"evaluate: LPIPS ready in {time.perf_counter() - t_lm:.1f}s", flush=True)
    except Exception as exc:
        lpips_model = None
        print(f"evaluate: LPIPS load failed ({exc!r}); metric stage may error.", flush=True)

    aesthetics_models = None
    if not skip_aesthetics_metrics:
        try:
            print(
                "evaluate: loading aesthetics models (CLIP ViT-H-14 + scorer heads; can take several minutes)…",
                flush=True,
            )
            t_am = time.perf_counter()
            from wmbench.metrics.aesthetics import load_aesthetics_and_artifacts_models

            aesthetics_models = load_aesthetics_and_artifacts_models(device=aesthetics_dev)
            print(f"evaluate: aesthetics models ready in {time.perf_counter() - t_am:.1f}s", flush=True)
        except Exception as exc:
            print(f"evaluate: aesthetics load failed ({exc!r}); aesthetics/artifacts will be NaN.", flush=True)

    print("evaluate: preparing FID reference mirror…", flush=True)
    fid_ref_mirror = _ensure_fid_reference_mirror(work_dir, originals_dir)
    print("evaluate: starting metric phases (similarity+LPIPS → FID → CLIP-FID → aesthetics → write)…", flush=True)

    # Phase 1 — PSNR / SSIM / NMI / LPIPS (reload pairs per cell; keeps GPU model hot).
    for key in tqdm(job_keys, desc="evaluate/similarity+LPIPS", unit="cell"):
        job = jobs[key]
        attack_name = job["attack_name"]
        stren_tag = job["stren_tag"]
        attacked_dir = job["attacked_dir"]
        originals, attacked, paired_basenames = _load_pairs(attacked_dir, originals_dir)
        job["paired_basenames"] = paired_basenames

        row: dict[str, float] = {"P": float(job["p_det"])}
        if not originals:
            for mk in agg.METRIC_KEYS:
                row[mk] = float("nan")
            job["row"] = row
            job["has_pairs"] = False
            continue

        job["has_pairs"] = True
        try:
            row["PSNR"] = compute_psnr(attacked, originals)
        except Exception:
            row["PSNR"] = float("nan")
        try:
            row["SSIM"] = compute_ssim(attacked, originals)
        except Exception:
            row["SSIM"] = float("nan")
        try:
            row["NMI"] = compute_nmi(attacked, originals)
        except Exception:
            row["NMI"] = float("nan")
        try:
            row["LPIPS"] = _require_finite(
                "LPIPS",
                compute_lpips(
                    attacked,
                    originals,
                    device=dev,
                    model=lpips_model,
                    verbose=metric_progress,
                    batch_size=lpips_batch_size,
                ),
                attack_name,
                stren_tag,
            )
        except Exception as e:
            raise RuntimeError(
                f"Required metric LPIPS failed for attack={attack_name}, strength={stren_tag}"
            ) from e
        job["row"] = row

    # Phase 2 — Inception FID (all cells).
    for key in tqdm(job_keys, desc="evaluate/FID(Inception)", unit="cell"):
        job = jobs[key]
        if not job.get("has_pairs"):
            continue
        attack_name = job["attack_name"]
        stren_tag = job["stren_tag"]
        attacked_dir = job["attacked_dir"]
        row = job["row"]
        assert row is not None
        try:
            if fid_ref_mirror:
                fid_value = compute_fid_from_dirs(
                    fid_ref_mirror,
                    attacked_dir,
                    device=dev,
                    verbose=metric_progress,
                )
            else:
                originals, attacked, _ = _load_pairs(attacked_dir, originals_dir)
                fid_value = compute_fid_metric(attacked, originals, device=dev, verbose=metric_progress)
            row["FID"] = _require_finite("FID", fid_value, attack_name, stren_tag)
        except Exception as e:
            raise RuntimeError(
                f"Required metric FID failed for attack={attack_name}, strength={stren_tag}"
            ) from e

    # Phase 3 — CLIP-FID (all cells).
    for key in tqdm(job_keys, desc="evaluate/CLIP-FID", unit="cell"):
        job = jobs[key]
        if not job.get("has_pairs"):
            continue
        attack_name = job["attack_name"]
        stren_tag = job["stren_tag"]
        attacked_dir = job["attacked_dir"]
        row = job["row"]
        assert row is not None
        try:
            if fid_ref_mirror:
                clip_fid_value = compute_clip_fid_from_dirs(
                    fid_ref_mirror,
                    attacked_dir,
                    device=dev,
                    verbose=metric_progress,
                )
            else:
                originals, attacked, _ = _load_pairs(attacked_dir, originals_dir)
                clip_fid_value = compute_clip_fid(
                    attacked,
                    originals,
                    device=dev,
                    verbose=metric_progress,
                )
            row["CLIP_FID"] = _require_finite("CLIP_FID", clip_fid_value, attack_name, stren_tag)
        except Exception as e:
            raise RuntimeError(
                f"Required metric CLIP_FID failed for attack={attack_name}, strength={stren_tag}"
            ) from e

    # Phase 4 — aesthetics / artifacts (all cells).
    for key in tqdm(job_keys, desc="evaluate/aesthetics", unit="cell"):
        job = jobs[key]
        row = job["row"]
        assert row is not None
        if not job.get("has_pairs"):
            continue
        attack_name = job["attack_name"]
        stren_tag = job["stren_tag"]
        attacked_dir = job["attacked_dir"]
        originals, attacked, paired_basenames = _load_pairs(attacked_dir, originals_dir)

        if aesthetics_models is not None:
            try:
                print(f"aesthetics/{attack_name}/{stren_tag}: start", flush=True)
                t0 = time.perf_counter()
                basename_key = tuple(paired_basenames)
                original_ratings = aesthetics_orig_rating_cache.get(basename_key)
                if original_ratings is None:
                    original_ratings, _ = compute_aesthetics_and_artifacts_scores(
                        originals, aesthetics_models, device=aesthetics_dev
                    )
                    aesthetics_orig_rating_cache[basename_key] = original_ratings
                row["aesthetics_delta"], row["artifacts"] = compute_aesthetics_delta_and_artifacts(
                    attacked,
                    originals,
                    device=aesthetics_dev,
                    aesthetics_models=aesthetics_models,
                    original_ratings=original_ratings,
                )
                dt = time.perf_counter() - t0
                print(f"aesthetics/{attack_name}/{stren_tag}: done in {dt:.2f}s", flush=True)
            except Exception:
                row["aesthetics_delta"] = float("nan")
                row["artifacts"] = float("nan")
        else:
            row["aesthetics_delta"] = float("nan")
            row["artifacts"] = float("nan")
            if not aesthetics_logged:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "missing_components.txt"), "a", encoding="utf-8") as mf:
                    mf.write("aesthetics/artifacts models unavailable\n")
                aesthetics_logged = True

    # Phase 5 — write metrics.json + .done
    for key in job_keys:
        job = jobs[key]
        row = job["row"]
        assert row is not None
        out_json = os.path.join(job["out_dir"], "metrics.json")
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(row, jf, indent=2)
        mark_done(job["done_flag"])

    if profile_metrics:
        dt = time.perf_counter() - stage_t0
        if dev.type == "cuda":
            try:
                peak_mb = torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
                print(
                    f"evaluate: profile total={dt:.1f}s, peak_cuda_alloc={peak_mb:.1f}MB, "
                    f"lpips_batch_size={lpips_batch_size}",
                    flush=True,
                )
            except Exception:
                print(f"evaluate: profile total={dt:.1f}s, lpips_batch_size={lpips_batch_size}", flush=True)
        else:
            print(f"evaluate: profile total={dt:.1f}s, lpips_batch_size={lpips_batch_size}", flush=True)
