from __future__ import annotations

"""DWT watermark adapter that uses implementation under D:\\waves\\dwt."""

import importlib.util
import os

import numpy as np
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _load_dwt_module():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dwt_impl_path = os.path.join(root, "dwt", "dwt_watermark_xia1998.py")
    if not os.path.isfile(dwt_impl_path):
        raise FileNotFoundError(f"Missing DWT implementation file: {dwt_impl_path}")
    spec = importlib.util.spec_from_file_location("wmbench_external_dwt_impl", dwt_impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for DWT implementation: {dwt_impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DWTAdapter(WatermarkAdapter):
    def __init__(
        self,
        alpha: float = 0.04,
        levels: int = 2,
        wavelet: str = "haar",
        seed: int = 1234,
        largest_fraction: float = 0.10,
        ratio_threshold: float = 1.05,
    ):
        self.alpha = alpha
        self.levels = levels
        self.wavelet = wavelet
        self.seed = seed
        self.largest_fraction = largest_fraction
        self.ratio_threshold = ratio_threshold
        self._last_payload: dict | None = None
        self._dwt_impl = _load_dwt_module()

    @property
    def name(self) -> str:
        return "dwt"

    def embed(self, image: Image.Image) -> Image.Image:
        img = image.convert("L")
        arr = img_as_float_or_uint8(np.asarray(img))
        x_hat, payload = self._dwt_impl.embed_watermark_dwt(
            arr,
            alpha=self.alpha,
            levels=self.levels,
            wavelet=self.wavelet,
            seed=self.seed,
            largest_fraction=self.largest_fraction,
        )
        self._last_payload = payload
        if float(x_hat.max()) <= 1.0 + 1e-6:
            x_hat_uint8 = np.clip(np.rint(x_hat * 255.0), 0, 255).astype(np.uint8)
        else:
            x_hat_uint8 = np.clip(np.rint(x_hat), 0, 255).astype(np.uint8)
        return Image.fromarray(x_hat_uint8).convert("RGB")

    def payload_for_meta(self) -> dict | None:
        return self._last_payload

    def detect(self, image: Image.Image, original: Image.Image, *, meta: dict | None = None) -> float:
        payload = meta
        if payload is None:
            payload = self._last_payload
        if payload is None:
            raise RuntimeError("DWTAdapter.detect requires meta (payload) from embed sidecar")
        o = np.asarray(original.convert("L"), dtype=float)
        c = np.asarray(image.convert("L"), dtype=float)
        if c.shape != o.shape:
            pil_c = Image.fromarray(c.astype(np.uint8) if c.max() > 1 else (c * 255).astype(np.uint8))
            pil_c = pil_c.resize((o.shape[1], o.shape[0]), Image.Resampling.BICUBIC)
            c = np.asarray(pil_c, dtype=float)
        if c.max() <= 1.0 + 1e-6:
            c = c * 255.0
        if o.max() <= 1.0 + 1e-6:
            o = o * 255.0
        _ok, _rec, all_records = self._dwt_impl.detect_watermark_hierarchical(
            o, c, payload, ratio_threshold=self.ratio_threshold
        )
        ratios = [r["mean_peak_ratio"] for r in all_records]
        max_ratio = max(ratios) if ratios else 0.0
        return float(np.clip((max_ratio - 1.0) / max(self.ratio_threshold - 1.0, 1e-6), 0.0, 1.0))


def img_as_float_or_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    return arr.astype(np.float64)
