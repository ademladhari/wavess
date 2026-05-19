from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from wmbench.pipeline.embed import meta_sidecar_path
from wmbench.pipeline.resume import is_done, mark_done
from wmbench.watermarks.base import WatermarkAdapter


def _remove_truncated_attacked_images(paths: list[str], append_log: str) -> list[str]:
    """Drop unreadable images (e.g. interrupted PNG writes); delete files and log paths."""
    removed: list[str] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.load()
                im.convert("RGB")
        except OSError:
            removed.append(p)
            try:
                os.unlink(p)
            except OSError:
                pass
    if removed:
        parent = os.path.dirname(append_log)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(append_log, "a", encoding="utf-8") as lf:
            for p in removed:
                lf.write(p + "\n")
    return removed


def _sidecar_embed_meta(method: str, sidecar: dict) -> dict | None:
    if method == "dct":
        return sidecar.get("dct_embed")
    if method == "dwt":
        return sidecar.get("dwt_payload")
    return None


def tpr_at_fpr(positive: np.ndarray, negative: np.ndarray, fpr_target: float = 0.001) -> float:
    """TPR at threshold = (1 - fpr_target) quantile of negative scores (99.9th pct for 0.1% FPR)."""
    if negative.size == 0:
        raise ValueError("negative scores required for TPR@FPR")
    thr = float(np.quantile(negative, 1.0 - fpr_target))
    if positive.size == 0:
        return 0.0
    return float(np.mean(positive > thr))


def run_detect_stage(
    adapter: WatermarkAdapter,
    work_dir: str,
    originals_dir: str,
    negatives_dir: str,
    attack_names: list[str],
    strength_values: dict[str, list],
    *,
    resume: bool = False,
    blind_detect: bool = False,
) -> None:
    watermarked_dir = os.path.join(work_dir, "watermarked")
    attacked_root = os.path.join(work_dir, "attacked")
    scores_root = os.path.join(work_dir, "scores")

    neg_paths = sorted(
        p
        for p in glob.glob(os.path.join(negatives_dir, "*"))
        if os.path.isfile(p) and p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    )
    neg_scores: list[float] = []
    for p in tqdm(neg_paths, desc=f"neg/{adapter.name}"):
        with Image.open(p) as im:
            neg = im.convert("RGB")
        if blind_detect:
            neg_scores.append(float(adapter.detect(neg, None, meta=None, blind=True)))
        else:
            orig_path = os.path.join(originals_dir, os.path.basename(p))
            if not os.path.isfile(orig_path):
                raise FileNotFoundError(
                    "Negative image does not have a matching original by basename for calibration: "
                    f"{os.path.basename(p)!r} (expected original at {orig_path})"
                )
            with Image.open(orig_path) as oi:
                orig = oi.convert("RGB")
            neg_scores.append(float(adapter.detect(neg, orig, meta=None, blind=False)))

    for attack_name in attack_names:
        for strength in strength_values.get(attack_name, []):
            stren_tag = str(strength).replace(os.sep, "_")
            attacked_dir = os.path.join(attacked_root, attack_name, stren_tag)
            out_dir = os.path.join(scores_root, attack_name, stren_tag)
            done_flag = os.path.join(out_dir, ".done")
            if resume and is_done(done_flag):
                continue
            os.makedirs(out_dir, exist_ok=True)
            atk_paths = sorted(
                p
                for p in glob.glob(os.path.join(attacked_dir, "*"))
                if os.path.isfile(p)
                and p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
            )
            corrupt_log = os.path.join(work_dir, "truncated_attacked_removed.log")
            removed = _remove_truncated_attacked_images(atk_paths, corrupt_log)
            if removed:
                attack_done = os.path.join(attacked_dir, ".done")
                for flag in (attack_done, done_flag):
                    if os.path.isfile(flag):
                        os.unlink(flag)
                partial_scores = os.path.join(out_dir, "scores.json")
                if os.path.isfile(partial_scores):
                    os.unlink(partial_scores)
                names = ", ".join(os.path.basename(p) for p in removed[:20])
                more = "" if len(removed) <= 20 else f" (+{len(removed) - 20} more)"
                raise RuntimeError(
                    "Removed unreadable attacked image(s) (often truncated after stopping the run mid-save). "
                    f"Deleted {len(removed)} file(s); logged paths to {corrupt_log}. Examples: {names}{more}. "
                    "Re-run with --resume to regenerate missing attacks and scores."
                )
            pos_scores: list[float] = []
            for ap in tqdm(atk_paths, desc=f"scores/{attack_name}/{stren_tag}"):
                base = os.path.basename(ap)
                wm_src = os.path.join(watermarked_dir, base)
                with Image.open(ap) as im:
                    att = im.convert("RGB")
                embed_meta = None
                sc = meta_sidecar_path(wm_src)
                if os.path.isfile(sc):
                    with open(sc, "rb") as mf:
                        sidecar = pickle.load(mf)
                    embed_meta = _sidecar_embed_meta(adapter.name, sidecar)
                if blind_detect:
                    pos_scores.append(float(adapter.detect(att, None, meta=embed_meta, blind=True)))
                else:
                    orig_path = os.path.join(originals_dir, base)
                    with Image.open(orig_path) as oi:
                        oimg = oi.convert("RGB")
                    pos_scores.append(float(adapter.detect(att, oimg, meta=embed_meta, blind=False)))

            out_json = os.path.join(out_dir, "scores.json")
            with open(out_json, "w", encoding="utf-8") as jf:
                json.dump({"positive": pos_scores, "negative": neg_scores}, jf)
            mark_done(done_flag)
