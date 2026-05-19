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


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _list_negative_image_paths(negatives_dir: str) -> list[str]:
    """Top-level images first; if none, search recursively (e.g. Drive layout negative_500/images/)."""
    root = os.path.abspath(negatives_dir)
    top = sorted(
        p
        for p in glob.glob(os.path.join(root, "*"))
        if os.path.isfile(p) and p.lower().endswith(_IMG_EXTS)
    )
    if top:
        return top
    deep = sorted(
        p
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(_IMG_EXTS) and ".wmbench_meta" not in p
    )
    return deep


def _sidecar_embed_meta(method: str, sidecar: dict) -> dict | None:
    if method == "dct":
        return sidecar.get("dct_embed")
    if method == "dwt":
        return sidecar.get("dwt_payload")
    if method in ("dct-dwt", "dct_dwt", "dctdwt"):
        return sidecar.get("dct_dwt_payload")
    if method == "svd":
        return sidecar.get("svd_payload")
    return None


def _list_svd_payload_bank(watermarked_dir: str) -> list[dict]:
    payloads: list[dict] = []
    wm_paths = sorted(
        p
        for p in glob.glob(os.path.join(watermarked_dir, "*"))
        if os.path.isfile(p) and p.lower().endswith(_IMG_EXTS) and ".wmbench_meta" not in p
    )
    for wp in wm_paths:
        sc = meta_sidecar_path(wp)
        if not os.path.isfile(sc):
            continue
        with open(sc, "rb") as mf:
            sidecar = pickle.load(mf)
        pl = sidecar.get("svd_payload")
        if pl is not None:
            payloads.append(pl)
    return payloads


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

    neg_paths = _list_negative_image_paths(negatives_dir)
    if not neg_paths:
        raise FileNotFoundError(
            "No negative calibration images found under --negatives. "
            f"Expected *.png/*.jpg/*.jpeg/*.webp/*.bmp under {os.path.abspath(negatives_dir)!r} "
            "(top-level or any subfolder). Blind detection still needs negatives to set the TPR@FPR threshold."
        )
    svd_neg_payload_bank: list[dict] = []
    if blind_detect and adapter.name == "svd":
        svd_neg_payload_bank = _list_svd_payload_bank(watermarked_dir)
        if not svd_neg_payload_bank:
            raise RuntimeError(
                "SVD blind detection needs key payloads from watermarked sidecars, but none were found. "
                "Re-run embed for method svd so .wmbench_meta.pkl includes svd_payload."
            )
    neg_scores: list[float] = []
    for i, p in enumerate(tqdm(neg_paths, desc=f"neg/{adapter.name}")):
        with Image.open(p) as im:
            neg = im.convert("RGB")
        if blind_detect:
            if adapter.name == "svd":
                key_meta = svd_neg_payload_bank[i % len(svd_neg_payload_bank)]
                neg_scores.append(float(adapter.detect(neg, None, meta=key_meta, blind=True)))
            else:
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
