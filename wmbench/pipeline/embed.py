from __future__ import annotations

import glob
import os
import pickle

from PIL import Image
from tqdm.auto import tqdm

from wmbench.pipeline.fsutil import atomic_image_save
from wmbench.pipeline.resume import is_done, mark_done
from wmbench.watermarks.base import WatermarkAdapter


def meta_sidecar_path(image_path: str) -> str:
    base, _ = os.path.splitext(image_path)
    return base + ".wmbench_meta.pkl"


def run_embed(
    adapter: WatermarkAdapter,
    image_paths: list[str],
    out_dir: str,
    *,
    resume: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, ".done")
    if resume and is_done(done_flag):
        return

    for p in tqdm(image_paths, desc=f"embed/{adapter.name}"):
        base = os.path.basename(p)
        dest = os.path.join(out_dir, base)
        if resume and os.path.isfile(dest):
            continue
        with Image.open(p) as im:
            wm = adapter.embed(im.convert("RGB"))
        atomic_image_save(wm, dest)
        meta = {"method": adapter.name}
        if hasattr(adapter, "payload_for_meta"):
            pl = adapter.payload_for_meta()
            if pl is not None:
                if adapter.name == "dct":
                    meta["dct_embed"] = pl
                elif adapter.name == "dwt":
                    meta["dwt_payload"] = pl
                elif adapter.name == "dct-dwt":
                    meta["dct_dwt_payload"] = pl
                elif adapter.name == "svd":
                    meta["svd_payload"] = pl
        with open(meta_sidecar_path(dest), "wb") as mf:
            pickle.dump(meta, mf, protocol=4)

    mark_done(done_flag)


def list_image_paths(directory: str) -> list[str]:
    paths: list[str] = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        paths.extend(glob.glob(os.path.join(directory, pat)))
        paths.extend(glob.glob(os.path.join(directory, pat.upper())))
    return sorted(set(paths))
