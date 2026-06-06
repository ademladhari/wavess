from __future__ import annotations

import glob
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm.auto import tqdm

from wmbench.pipeline.fsutil import atomic_image_save
from wmbench.pipeline.resume import is_done, mark_done
from wmbench.watermarks.base import WatermarkAdapter


def meta_sidecar_path(image_path: str) -> str:
    base, _ = os.path.splitext(image_path)
    return base + ".wmbench_meta.pkl"


def _embed_meta_for_adapter(adapter: WatermarkAdapter) -> dict:
    meta: dict = {"method": adapter.name}
    if hasattr(adapter, "payload_for_meta"):
        pl = adapter.payload_for_meta()
        if pl is not None:
            if adapter.name == "dct":
                meta["dct_embed"] = pl
            elif adapter.name == "dwt":
                meta["dwt_payload"] = pl
            elif adapter.name == "dct-dwt":
                meta["dct_dwt_payload"] = pl
            elif adapter.name == "dwt-dct-svd":
                meta["dwt_dct_svd_payload"] = pl
            elif adapter.name == "svd":
                meta["svd_payload"] = pl
            elif adapter.name == "flexible":
                meta["flexible_payload"] = pl
            elif adapter.name == "ssl":
                meta["ssl_payload"] = pl
            elif adapter.name == "tree-ring":
                meta["tree_ring_payload"] = pl
    return meta


def _write_embed_row(dest: str, wm: Image.Image, meta: dict) -> None:
    atomic_image_save(wm, dest)
    with open(meta_sidecar_path(dest), "wb") as mf:
        pickle.dump(meta, mf, protocol=4)


def _embed_batch_cpu_parallel(
    adapter: WatermarkAdapter,
    images: list[Image.Image],
    *,
    max_workers: int,
) -> list[tuple[Image.Image, dict]]:
    """Parallel CPU embed with per-image sidecar metadata."""

    def _one(im: Image.Image) -> tuple[Image.Image, dict]:
        wm = adapter.embed(im)
        return wm, _embed_meta_for_adapter(adapter)

    workers = max(1, min(len(images), max_workers))
    if workers == 1:
        return [_one(im) for im in images]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, images))


def run_embed(
    adapter: WatermarkAdapter,
    image_paths: list[str],
    out_dir: str,
    *,
    resume: bool = False,
    embed_batch_size: int = 1,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, ".done")
    if resume and is_done(done_flag):
        return

    embed_batch_size = max(1, int(embed_batch_size))
    cpu_workers = max(1, min(embed_batch_size, os.cpu_count() or 4))

    pending: list[str] = []
    for p in image_paths:
        base = os.path.basename(p)
        dest = os.path.join(out_dir, base)
        if resume and os.path.isfile(dest) and os.path.isfile(meta_sidecar_path(dest)):
            continue
        pending.append(p)

    if not pending:
        mark_done(done_flag)
        return

    shared_meta = bool(getattr(adapter, "embed_meta_shared", False))

    with tqdm(total=len(pending), desc=f"embed/{adapter.name}", unit="img") as pbar:
        i = 0
        while i < len(pending):
            batch_paths = pending[i : i + embed_batch_size]
            images: list[Image.Image] = []
            for src in batch_paths:
                with Image.open(src) as im:
                    images.append(im.convert("RGB"))

            if shared_meta:
                wm_images = adapter.embed_batch(images)
                if len(wm_images) != len(batch_paths):
                    raise RuntimeError(
                        f"embed batch output length mismatch: expected {len(batch_paths)}, "
                        f"got {len(wm_images)}"
                    )
                meta = _embed_meta_for_adapter(adapter)
                for src, wm in zip(batch_paths, wm_images):
                    dest = os.path.join(out_dir, os.path.basename(src))
                    _write_embed_row(dest, wm, meta)
            else:
                rows = _embed_batch_cpu_parallel(adapter, images, max_workers=cpu_workers)
                if len(rows) != len(batch_paths):
                    raise RuntimeError(
                        f"embed batch output length mismatch: expected {len(batch_paths)}, "
                        f"got {len(rows)}"
                    )
                for src, (wm, meta) in zip(batch_paths, rows):
                    dest = os.path.join(out_dir, os.path.basename(src))
                    _write_embed_row(dest, wm, meta)

            i += len(batch_paths)
            pbar.update(len(batch_paths))

    mark_done(done_flag)


def list_image_paths(directory: str) -> list[str]:
    paths: list[str] = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        paths.extend(glob.glob(os.path.join(directory, pat)))
        paths.extend(glob.glob(os.path.join(directory, pat.upper())))
    return sorted(set(paths))
