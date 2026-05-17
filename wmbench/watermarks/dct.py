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
        return Image.fromarray(np.clip(wm.image, 0, 255).astype(np.uint8)).convert("RGB")

    def detect(self, image: Image.Image, original: Image.Image, *, meta: dict | None = None) -> float:
        del meta
        o = original.convert("L")
        if image.mode != "L":
            image = image.convert("RGB")
        c = image.convert("L")
        if c.size != o.size:
            c = c.resize(o.size, Image.Resampling.BICUBIC)
        o_arr = np.asarray(o, dtype=np.float64)
        c_arr = np.asarray(c, dtype=np.float64)
        rows, cols = self._dct_impl.top_magnitude_indices(self._dct_impl.dct2(o_arr), self._bit_length)
        extracted = self._dct_impl.extract_watermark(
            original_image=o_arr,
            candidate_image=c_arr,
            rows=rows,
            cols=cols,
            alpha=self._alpha,
        )
        return float(self._dct_impl.similarity(self._mark, extracted))
