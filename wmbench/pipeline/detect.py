from __future__ import annotations

import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from wmbench.pipeline.embed import meta_sidecar_path
from wmbench.pipeline.resume import is_done, mark_done
from wmbench.watermarks.base import WatermarkAdapter


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_TREE_RING_WORKER_ADAPTER = None
_TREE_RING_WORKER_DEVICE = None


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
    if method in ("dwt-dct-svd", "dwt_dct_svd", "dwtdctsvd"):
        return sidecar.get("dwt_dct_svd_payload")
    if method == "svd":
        return sidecar.get("svd_payload")
    if method in ("flexible", "flex"):
        return sidecar.get("flexible_payload")
    if method in ("ssl", "ssl-wm", "ssl_wm"):
        return sidecar.get("ssl_payload")
    if method in ("tree-ring", "tree_ring", "treering"):
        return sidecar.get("tree_ring_payload")
    return None


def _list_payload_bank(watermarked_dir: str, payload_key: str) -> list[dict]:
    """Collect embed payloads from watermarked sidecars (for blind SVD or non-blind dwt-dct-svd negatives)."""
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
        pl = sidecar.get(payload_key)
        if pl is not None:
            payloads.append(pl)
    return payloads


def _list_svd_payload_bank(watermarked_dir: str) -> list[dict]:
    return _list_payload_bank(watermarked_dir, "svd_payload")


def _file_state_key(paths: list[str]) -> str:
    """Stable cache key from path + mtime + size."""
    rows: list[str] = []
    for p in paths:
        try:
            st = os.stat(p)
            rows.append(f"{os.path.abspath(p)}|{int(st.st_size)}|{int(st.st_mtime_ns)}")
        except OSError:
            rows.append(f"{os.path.abspath(p)}|missing")
    h = hashlib.sha1("\n".join(rows).encode("utf-8"), usedforsecurity=False).hexdigest()
    return h


def _neg_cache_path(scores_root: str, method: str, blind_detect: bool) -> str:
    mode = "blind" if blind_detect else "nonblind"
    safe_method = method.replace(os.sep, "_")
    return os.path.join(scores_root, f"_neg_cache_{safe_method}_{mode}.json")


def _tree_ring_detect_gpu_ids() -> list[str]:
    """
    Optional multi-GPU override for Tree-Ring detect path.
    Example: WMBENCH_TREE_RING_DETECT_GPUS=0,1
    """
    raw = os.environ.get("WMBENCH_TREE_RING_DETECT_GPUS", "").strip()
    if not raw:
        return []
    out: list[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.lower().startswith("cuda:"):
            t = t.split(":", 1)[1].strip()
        if t and t not in out:
            out.append(t)
    return out


def _tree_ring_detect_worker(
    items: list[tuple[int, str]],
    *,
    blind_detect: bool,
    originals_dir: str,
    visible_gpu: str,
) -> list[tuple[int, float]]:
    global _TREE_RING_WORKER_ADAPTER, _TREE_RING_WORKER_DEVICE

    # Hard-pin this worker to a concrete CUDA device id.
    device = f"cuda:{visible_gpu}"
    os.environ["WMBENCH_TREE_RING_DEVICE"] = device
    os.environ["WMBENCH_TREE_RING_INVERT_DEVICE"] = device

    if _TREE_RING_WORKER_ADAPTER is None or _TREE_RING_WORKER_DEVICE != device:
        from wmbench.watermarks import get_adapter

        _TREE_RING_WORKER_ADAPTER = get_adapter("tree-ring")
        _TREE_RING_WORKER_DEVICE = device

    adapter = _TREE_RING_WORKER_ADAPTER
    scores: list[tuple[int, float]] = []
    for idx, p in items:
        with Image.open(p) as im:
            img = im.convert("RGB")
        if blind_detect:
            sc = float(adapter.detect(img, None, meta=None, blind=True))
        else:
            orig_path = os.path.join(originals_dir, os.path.basename(p))
            with Image.open(orig_path) as oi:
                oimg = oi.convert("RGB")
            sc = float(adapter.detect(img, oimg, meta=None, blind=False))
        scores.append((idx, sc))
    return scores


def _tree_ring_score_paths_multi_gpu(
    paths: list[str],
    *,
    blind_detect: bool,
    originals_dir: str,
    gpu_ids: list[str],
    desc: str,
) -> list[float]:
    """
    Score paths with Tree-Ring detector across multiple GPUs.
    Returns scores in the same order as `paths`.
    """
    if not paths:
        return []
    if len(gpu_ids) < 2:
        raise ValueError("multi-gpu scoring requires at least 2 GPU ids")

    chunk_size = max(1, int(os.environ.get("WMBENCH_TREE_RING_DETECT_CHUNK", "16") or "16"))
    indexed_paths = list(enumerate(paths))
    chunks: list[tuple[str, list[tuple[int, str]]]] = []
    for start in range(0, len(indexed_paths), chunk_size):
        chunk = indexed_paths[start : start + chunk_size]
        gid = gpu_ids[(start // chunk_size) % len(gpu_ids)]
        chunks.append((gid, chunk))

    scores = [0.0] * len(paths)
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp_ctx) as ex:
        futs = [
            ex.submit(
                _tree_ring_detect_worker,
                chunk,
                blind_detect=blind_detect,
                originals_dir=originals_dir,
                visible_gpu=gid,
            )
            for gid, chunk in chunks
        ]
        with tqdm(total=len(paths), desc=desc, unit="img") as pbar:
            for fut in as_completed(futs):
                batch = fut.result()
                for idx, sc in batch:
                    scores[idx] = float(sc)
                pbar.update(len(batch))

    return scores


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
    dwt_dct_svd_neg_payload_bank: list[dict] = []
    if not blind_detect and adapter.name == "dwt-dct-svd":
        dwt_dct_svd_neg_payload_bank = _list_payload_bank(watermarked_dir, "dwt_dct_svd_payload")
        if not dwt_dct_svd_neg_payload_bank:
            raise RuntimeError(
                "dwt-dct-svd non-blind detection needs embed sidecars (dwt_dct_svd_payload) under "
                f"{watermarked_dir}. Re-run embed so each watermarked image has .wmbench_meta.pkl."
            )
    neg_cache_key = {
        "adapter": adapter.name,
        "blind_detect": bool(blind_detect),
        "negatives_dir": os.path.abspath(negatives_dir),
        "neg_paths_state": _file_state_key(neg_paths),
    }
    if blind_detect and adapter.name == "svd":
        neg_cache_key["svd_bank_size"] = len(svd_neg_payload_bank)
    if (not blind_detect) and adapter.name == "dwt-dct-svd":
        neg_cache_key["dwt_dct_svd_bank_size"] = len(dwt_dct_svd_neg_payload_bank)
    neg_cache_sig = hashlib.sha1(
        json.dumps(neg_cache_key, sort_keys=True).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    cache_path = _neg_cache_path(scores_root, adapter.name, blind_detect)
    neg_scores: list[float] = []
    cache_loaded = False
    tree_ring_gpus = _tree_ring_detect_gpu_ids() if adapter.name == "tree-ring" else []
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as cf:
                cached = json.load(cf)
            if cached.get("sig") == neg_cache_sig and isinstance(cached.get("scores"), list):
                neg_scores = [float(x) for x in cached["scores"]]
                cache_loaded = True
        except Exception:
            cache_loaded = False
    if not cache_loaded:
        used_tree_ring_mgpu = False
        if adapter.name == "tree-ring" and len(tree_ring_gpus) > 1:
            try:
                print(
                    f"tree-ring detect multi-GPU enabled for neg cache: {tree_ring_gpus}",
                    flush=True,
                )
                neg_scores = _tree_ring_score_paths_multi_gpu(
                    neg_paths,
                    blind_detect=blind_detect,
                    originals_dir=originals_dir,
                    gpu_ids=tree_ring_gpus,
                    desc=f"neg/{adapter.name}",
                )
                used_tree_ring_mgpu = True
            except Exception as e:
                print(
                    f"tree-ring multi-GPU neg detect failed ({e!r}); falling back to single-process detect.",
                    flush=True,
                )
                neg_scores = []
        if not used_tree_ring_mgpu:
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
                    neg_meta = None
                    if adapter.name == "dwt-dct-svd":
                        neg_meta = dwt_dct_svd_neg_payload_bank[i % len(dwt_dct_svd_neg_payload_bank)]
                    neg_scores.append(float(adapter.detect(neg, orig, meta=neg_meta, blind=False)))
        try:
            os.makedirs(scores_root, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as cf:
                json.dump({"sig": neg_cache_sig, "scores": neg_scores}, cf)
        except OSError:
            pass

    # Sidecar payload lookup is reused across all strengths/attacks to reduce repeated disk I/O.
    embed_meta_by_base: dict[str, dict | None] = {}
    wm_paths = sorted(
        p
        for p in glob.glob(os.path.join(watermarked_dir, "*"))
        if os.path.isfile(p)
        and p.lower().endswith(_IMG_EXTS)
        and ".wmbench_meta" not in os.path.basename(p)
    )
    for wp in wm_paths:
        base = os.path.basename(wp)
        sc = meta_sidecar_path(wp)
        embed_meta = None
        if os.path.isfile(sc):
            with open(sc, "rb") as mf:
                sidecar = pickle.load(mf)
            embed_meta = _sidecar_embed_meta(adapter.name, sidecar)
        embed_meta_by_base[base] = embed_meta

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

            def _score_path(ap: str) -> float:
                base = os.path.basename(ap)
                embed_meta = embed_meta_by_base.get(base)
                with Image.open(ap) as im:
                    att = im.convert("RGB")
                if blind_detect:
                    return float(adapter.detect(att, None, meta=embed_meta, blind=True))
                orig_path = os.path.join(originals_dir, base)
                with Image.open(orig_path) as oi:
                    oimg = oi.convert("RGB")
                return float(adapter.detect(att, oimg, meta=embed_meta, blind=False))

            dwt_threads = int(os.environ.get("WMBENCH_DWT_DETECT_THREADS", "1") or "1")
            if adapter.name == "dwt" and dwt_threads > 1:
                with ThreadPoolExecutor(max_workers=dwt_threads) as ex:
                    it = ex.map(_score_path, atk_paths)
                    for sc in tqdm(it, total=len(atk_paths), desc=f"scores/{attack_name}/{stren_tag}"):
                        pos_scores.append(float(sc))
            elif adapter.name == "tree-ring" and len(tree_ring_gpus) > 1:
                try:
                    print(
                        f"tree-ring detect multi-GPU enabled for scores/{attack_name}/{stren_tag}: {tree_ring_gpus}",
                        flush=True,
                    )
                    pos_scores = _tree_ring_score_paths_multi_gpu(
                        atk_paths,
                        blind_detect=blind_detect,
                        originals_dir=originals_dir,
                        gpu_ids=tree_ring_gpus,
                        desc=f"scores/{attack_name}/{stren_tag}",
                    )
                except Exception as e:
                    print(
                        f"tree-ring multi-GPU scores failed ({e!r}); falling back to single-process detect.",
                        flush=True,
                    )
                    for ap in tqdm(atk_paths, desc=f"scores/{attack_name}/{stren_tag}"):
                        pos_scores.append(_score_path(ap))
            else:
                for ap in tqdm(atk_paths, desc=f"scores/{attack_name}/{stren_tag}"):
                    pos_scores.append(_score_path(ap))

            out_json = os.path.join(out_dir, "scores.json")
            with open(out_json, "w", encoding="utf-8") as jf:
                json.dump({"positive": pos_scores, "negative": neg_scores}, jf)
            mark_done(done_flag)
