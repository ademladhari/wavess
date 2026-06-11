"""Per-method profiles and bit-accuracy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter

# Keep in sync with wmbench.metrics.aggregate.GENERATION_BASED_METHODS (avoid FID import chain).
_GENERATION_BASED_METHODS = frozenset(
    {"flexible", "flex", "tree-ring", "tree_ring", "treering", "robin"}
)
from wmbench.watermarks.dct import _load_dct_module


@dataclass(frozen=True)
class MethodProfile:
    method_id: str
    generation_based: bool
    blind_detect: bool
    supports_bit_accuracy: bool
    notes: str = ""


def _canonical(method_id: str) -> str:
    return method_id.strip().lower().replace("_", "-")


def profile_for(method_id: str, *, blind_detect: bool | None = None) -> MethodProfile:
    mid = _canonical(method_id)
    gen = mid in _GENERATION_BASED_METHODS or mid.replace("-", "_") in _GENERATION_BASED_METHODS

    # Methods without meaningful multi-bit payload
    no_bits = {"tree-ring", "treering", "tree_ring", "robin", "dwt", "svd"}

    if mid in {"dct", "dwt", "dct-dwt", "dctdwt", "dct_dwt"}:
        blind = bool(blind_detect) if blind_detect is not None else False
    elif mid in {"svd", "dwt-dct-svd", "dwtdctsvd", "dwt_dct_svd"}:
        blind = True
    elif mid in {"flexible", "flex", "tree-ring", "treering", "tree_ring", "robin", "ssl"}:
        blind = True
    else:
        blind = bool(blind_detect) if blind_detect is not None else False

    return MethodProfile(
        method_id=mid,
        generation_based=gen,
        blind_detect=blind,
        supports_bit_accuracy=mid not in no_bits,
        notes="generation-based: embed may ignore host pixels" if gen else "",
    )


def payload_from_adapter(adapter: WatermarkAdapter) -> dict | None:
    fn = getattr(adapter, "payload_for_meta", None)
    if callable(fn):
        return fn()
    return None


def _dct_bit_accuracy(
    original: Image.Image,
    attacked: Image.Image,
    meta: dict,
    adapter: WatermarkAdapter,
) -> float:
    dct_impl = _load_dct_module()
    o_gray = np.asarray(original.convert("L"), dtype=np.float64)
    c_gray = np.asarray(attacked.convert("L"), dtype=np.float64)
    if c_gray.shape != o_gray.shape:
        c_pil = attacked.convert("L").resize(
            (o_gray.shape[1], o_gray.shape[0]), Image.Resampling.BICUBIC
        )
        c_gray = np.asarray(c_pil, dtype=np.float64)
    mark = np.asarray(meta["mark"], dtype=np.float64)
    rows = np.asarray(meta["rows"])
    cols = np.asarray(meta["cols"])
    extracted = dct_impl.extract_watermark(o_gray, c_gray, rows, cols, adapter._alpha)  # type: ignore[attr-defined]
    return float(np.mean((mark > 0) == (extracted > 0)))


def _dct_dwt_bit_accuracy(
    _original: Image.Image,
    attacked: Image.Image,
    meta: dict,
    adapter: WatermarkAdapter,
) -> float:
    host = adapter._load_host_from_image(attacked, f"bit_{id(attacked)}")  # type: ignore[attr-defined]
    ex_bits = adapter._impl.extract_watermark_bits(  # type: ignore[attr-defined]
        host,
        capacity=int(meta["capacity"]),
        seed=int(meta["seed"]),
        wavelet=str(meta["wavelet"]),
        subband_choice=str(meta["subband_choice"]),
        mid_pos=[tuple(x) for x in meta["mid_pos"]],
        pn_mode=str(meta["pn_mode"]),
    )
    ref_bits = np.asarray(meta["wm_bits"], dtype=np.uint8).flatten()[: len(ex_bits)]
    return float(np.mean(ref_bits == ex_bits.astype(np.uint8)))


def _flexible_bit_accuracy(
    _original: Image.Image,
    attacked: Image.Image,
    meta: dict,
    adapter: WatermarkAdapter,
) -> float:
    # FlexibleAdapter.detect returns mean bit accuracy for the payload.
    return float(adapter.detect(attacked, None, meta=meta, blind=True))


def _ssl_bit_accuracy(
    _original: Image.Image,
    attacked: Image.Image,
    meta: dict,
    adapter: WatermarkAdapter,
) -> float:
    if adapter._mode == "zero_bit":  # type: ignore[attr-defined]
        return float("nan")
    return float(adapter.detect(attacked, None, meta=meta, blind=True))


_BIT_FN: dict[str, Callable[..., float]] = {
    "dct": _dct_bit_accuracy,
    "dct-dwt": _dct_dwt_bit_accuracy,
    "dct_dwt": _dct_dwt_bit_accuracy,
    "dctdwt": _dct_dwt_bit_accuracy,
    "flexible": _flexible_bit_accuracy,
    "flex": _flexible_bit_accuracy,
    "ssl": _ssl_bit_accuracy,
}


def compute_bit_accuracy(
    adapter: WatermarkAdapter,
    original: Image.Image,
    attacked: Image.Image,
    meta: dict | None,
    profile: MethodProfile,
) -> float:
    if not profile.supports_bit_accuracy:
        return float("nan")
    if meta is None:
        return float("nan")
    fn = _BIT_FN.get(profile.method_id)
    if fn is None:
        return float("nan")
    try:
        return fn(original, attacked, meta, adapter)
    except Exception:
        return float("nan")


def detection_score(
    adapter: WatermarkAdapter,
    image: Image.Image,
    original: Image.Image | None,
    meta: dict | None,
    *,
    blind: bool,
) -> float:
    if blind:
        return float(adapter.detect(image, None, meta=meta, blind=True))
    if original is None:
        raise ValueError("non-blind detection requires original image")
    return float(adapter.detect(image, original, meta=meta, blind=False))


def negative_detection_score(
    adapter: WatermarkAdapter,
    clean: Image.Image,
    original: Image.Image,
    profile: MethodProfile,
    *,
    neg_meta: dict | None,
    svd_payload_bank: list[dict] | None,
    neg_index: int,
) -> float:
    mid = profile.method_id
    if profile.blind_detect:
        if mid == "svd" and svd_payload_bank:
            key_meta = svd_payload_bank[neg_index % len(svd_payload_bank)]
            return detection_score(adapter, clean, None, key_meta, blind=True)
        if mid in {"flexible", "flex", "dct-dwt", "dct_dwt", "dctdwt", "ssl"} and neg_meta is not None:
            return detection_score(adapter, clean, None, neg_meta, blind=True)
        return detection_score(adapter, clean, None, None, blind=True)
    return detection_score(adapter, clean, original, neg_meta, blind=False)
