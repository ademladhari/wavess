from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict

import numpy as np
from PIL import Image

from wmbench.watermarks.base import WatermarkAdapter


def _resolve_impl_path(user_path: str | None = None) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        user_path,
        os.environ.get("WMBENCH_DWT_DCT_SVD_IMPL"),
        os.path.join(root, "dwt-dct-svd", "dwt_dct_svd.py"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    listed = ", ".join(repr(c) for c in candidates if c)
    raise FileNotFoundError(
        "Could not locate dwt_dct_svd.py implementation. "
        f"Tried: {listed}. Set WMBENCH_DWT_DCT_SVD_IMPL."
    )


def _load_impl(path: str):
    module_name = "wmbench_external_dwt_dct_svd"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    required = [
        "to_float01",
        "embed_signature_in_biometric",
        "embed_tw_in_cover",
        "extract_tw_from_watermarked",
        "extract_signature_from_tw",
        "ncc",
        "WatermarkState",
    ]
    missing = [k for k in required if not hasattr(mod, k)]
    if missing:
        raise RuntimeError(f"dwt_dct_svd implementation missing required symbols: {missing}")
    return mod


def _state_to_payload(state) -> dict:
    data = asdict(state)
    return data


class DWTDCTSVDAdapter(WatermarkAdapter):
    """
    Adapter that directly uses dwt-dct-svd/dwt_dct_svd.py functions.

    This method is non-blind by design (paper sections 3.2-3.3 require original cover
    and embedding-side state). In wmbench blind mode is intentionally unsupported.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        beta: float = 0.02,
        signature_size: int = 128,
        biometric_size: int = 256,
        seed: int = 1234,
        impl_path: str | None = None,
    ):
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._impl = _load_impl(_resolve_impl_path(impl_path))
        rng = np.random.default_rng(seed)
        self._signature = (rng.integers(0, 2, size=(signature_size, signature_size), dtype=np.uint8) * 255).astype(
            np.uint8
        )
        # Biometric template kept fixed per run for reproducibility.
        self._biometric = rng.integers(0, 256, size=(biometric_size, biometric_size), dtype=np.uint8)
        self._last_payload: dict | None = None

    @property
    def name(self) -> str:
        return "dwt-dct-svd"

    def payload_for_meta(self) -> dict | None:
        return self._last_payload

    def embed(self, image: Image.Image) -> Image.Image:
        cover_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        cover = self._impl.to_float01(cover_rgb)
        TW, state = self._impl.embed_signature_in_biometric(self._biometric, self._signature, self._alpha)
        wm, state = self._impl.embed_tw_in_cover(cover, TW, self._beta, state=state)
        self._last_payload = _state_to_payload(state)
        arr = np.clip(np.rint(np.asarray(wm, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def detect(
        self,
        image: Image.Image,
        original: Image.Image | None = None,
        *,
        meta: dict | None = None,
        blind: bool = False,
    ) -> float:
        if blind:
            raise RuntimeError(
                "DWTDCTSVDAdapter is non-blind by design and requires original cover image at detect."
            )
        if original is None:
            raise ValueError("DWTDCTSVDAdapter.detect requires original cover image")
        payload = meta if meta is not None else self._last_payload
        if payload is None:
            raise RuntimeError("DWTDCTSVDAdapter.detect requires embed sidecar payload")

        # Recreate state from saved payload fields.
        state = self._impl.WatermarkState(**payload)
        wm = self._impl.to_float01(np.asarray(image.convert("RGB"), dtype=np.uint8))
        cov = self._impl.to_float01(np.asarray(original.convert("RGB"), dtype=np.uint8))
        tw_hat = self._impl.extract_tw_from_watermarked(wm, cov, self._beta, state)
        sig_hat = self._impl.extract_signature_from_tw(tw_hat, self._biometric, self._alpha, state)
        score = float(self._impl.ncc(self._signature, sig_hat))
        # Map NCC [-1,1] -> [0,1]
        return float(np.clip((score + 1.0) * 0.5, 0.0, 1.0))

