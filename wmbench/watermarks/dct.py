from __future__ import annotations

import importlib.util
import os

import numpy as np
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _load_dct_module():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dct_impl_path = os.path.join(root, "dct", "reproduce_dct_paper.py")
    if not os.path.isfile(dct_impl_path):
        raise FileNotFoundError(f"Missing DCT implementation file: {dct_impl_path}")
    spec = importlib.util.spec_from_file_location("wmbench_external_dct_impl", dct_impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for DCT implementation: {dct_impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_gray(image: Image.Image, ref_shape: tuple[int, int] | None = None) -> np.ndarray:
    if image.mode != "L":
        image = image.convert("RGB")
    c = image.convert("L")
    if ref_shape is not None and c.size != (ref_shape[1], ref_shape[0]):
        c = c.resize((ref_shape[1], ref_shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(c, dtype=np.float64)


class DCTAdapter(WatermarkAdapter):
    def __init__(
        self,
        bit_length: int = 1000,
        seed: int = 42,
        alpha: float = 0.1,
    ):
        self._bit_length = bit_length
        self._seed = seed
        self._alpha = alpha
        self._dct_impl = _load_dct_module()
        self._mark = self._dct_impl.gaussian_watermark(bit_length, np.random.default_rng(seed))
        self._last_embed_meta: dict | None = None

    @property
    def name(self) -> str:
        return "dct"

    def embed(self, image: Image.Image) -> Image.Image:
        img = image.convert("L")
        arr = np.asarray(img, dtype=np.float64)
        wm = self._dct_impl.embed_watermark(
            arr,
            n=self._bit_length,
            alpha=self._alpha,
            rng=np.random.default_rng(self._seed),
            watermark=self._mark,
        )
        coeffs = self._dct_impl.dct2(arr)
        ref_coeffs = coeffs[wm.rows, wm.cols].astype(np.float64)
        self._last_embed_meta = {
            "rows": wm.rows.astype(np.int64),
            "cols": wm.cols.astype(np.int64),
            "mark": wm.watermark.astype(np.float64),
            "ref_coeffs": ref_coeffs,
            "shape": (int(arr.shape[0]), int(arr.shape[1])),
        }
        return Image.fromarray(np.clip(wm.image, 0, 255).astype(np.uint8)).convert("RGB")

    def payload_for_meta(self) -> dict | None:
        return self._last_embed_meta

    def _score_from_embed_meta(self, candidate_arr: np.ndarray, embed_meta: dict) -> float:
        rows = np.asarray(embed_meta["rows"], dtype=np.int64)
        cols = np.asarray(embed_meta["cols"], dtype=np.int64)
        mark = np.asarray(embed_meta["mark"], dtype=np.float64)
        ref = np.asarray(embed_meta["ref_coeffs"], dtype=np.float64)
        c1 = self._dct_impl.dct2(candidate_arr)[rows, cols]
        eps = 1e-10
        denom = np.where(np.abs(ref) < eps, eps, ref)
        extracted = (c1 / denom - 1.0) / self._alpha
        return float(self._dct_impl.similarity(mark, extracted))

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        if blind:
            embed_meta = meta if meta is not None else self._last_embed_meta
            if embed_meta is None:
                c_arr = _candidate_gray(image)
                rows, cols = self._dct_impl.top_magnitude_indices(
                    self._dct_impl.dct2(c_arr), self._bit_length
                )
                extracted = self._dct_impl.extract_watermark(c_arr, c_arr, rows, cols, self._alpha)
                return float(self._dct_impl.similarity(self._mark, extracted))
            shape = embed_meta.get("shape")
            ref_shape = (int(shape[0]), int(shape[1])) if shape else None
            c_arr = _candidate_gray(image, ref_shape=ref_shape)
            return self._score_from_embed_meta(c_arr, embed_meta)

        if original is None:
            raise ValueError("DCTAdapter non-blind detect requires original image")
        o = original.convert("L")
        c_arr = _candidate_gray(image, ref_shape=(o.height, o.width))
        o_arr = np.asarray(o, dtype=np.float64)
        rows, cols = self._dct_impl.top_magnitude_indices(
            self._dct_impl.dct2(o_arr), self._bit_length
        )
        extracted = self._dct_impl.extract_watermark(
            original_image=o_arr,
            candidate_image=c_arr,
            rows=rows,
            cols=cols,
            alpha=self._alpha,
        )
        return float(self._dct_impl.similarity(self._mark, extracted))
