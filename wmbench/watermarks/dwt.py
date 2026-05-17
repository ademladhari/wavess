from __future__ import annotations

"""DWT watermark (Xia et al. 1998-style), ported from dwt/dwt_watermark_xia1998.ipynb (logic only)."""

import numpy as np
import pywt
from PIL import Image
from scipy import signal

from wmbench.watermarks.base import WatermarkAdapter


def dwt2_multilevel(image: np.ndarray, levels: int = 2, wavelet: str = "haar"):
    return pywt.wavedec2(image, wavelet=wavelet, level=levels)


def idwt2_multilevel(coeffs, wavelet: str = "haar"):
    return pywt.waverec2(coeffs, wavelet=wavelet)


def dynamic_range_clip(x_tilde: np.ndarray, x_original: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x_original))
    x_max = float(np.max(x_original))
    return np.clip(x_tilde, x_min, x_max)


def crop_like(arr: np.ndarray, reference: np.ndarray) -> np.ndarray:
    r, c = reference.shape
    return arr[:r, :c]


def embed_watermark_dwt(
    image: np.ndarray,
    alpha: float = 0.04,
    levels: int = 2,
    wavelet: str = "haar",
    seed: int = 1234,
    largest_fraction: float = 0.10,
):
    rng = np.random.default_rng(seed)
    coeffs = dwt2_multilevel(image, levels=levels, wavelet=wavelet)

    cA = coeffs[0]
    new_coeffs = [cA.copy()]
    watermark_noise = {}
    watermark_signal = {}
    watermark_mask = {}

    for lev_idx, detail_triplet in enumerate(coeffs[1:], start=1):
        cH, cV, cD = detail_triplet
        triplet_out = []
        for band_name, band in zip(["LH", "HL", "HH"], [cH, cV, cD]):
            N = rng.normal(loc=0.0, scale=1.0, size=band.shape)
            abs_band = np.abs(band)
            thr = np.quantile(abs_band, 1.0 - largest_fraction)
            mask = abs_band >= thr

            added = alpha * (band**2) * N
            band_tilde = band.copy()
            band_tilde[mask] = band_tilde[mask] + added[mask]

            triplet_out.append(band_tilde)
            watermark_noise[(lev_idx, band_name)] = N
            watermark_signal[(lev_idx, band_name)] = np.where(mask, added, 0.0)
            watermark_mask[(lev_idx, band_name)] = mask
        new_coeffs.append(tuple(triplet_out))

    x_tilde = idwt2_multilevel(new_coeffs, wavelet=wavelet)
    x_tilde = crop_like(x_tilde, image)
    x_hat = dynamic_range_clip(x_tilde, image)

    payload = {
        "levels": levels,
        "wavelet": wavelet,
        "alpha": alpha,
        "seed": seed,
        "watermark_noise": watermark_noise,
        "watermark_signal": watermark_signal,
        "watermark_mask": watermark_mask,
        "largest_fraction": largest_fraction,
    }
    return x_hat, payload


def normalized_2d_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12
    corr = signal.correlate2d(a0, b0, mode="full", boundary="fill", fillvalue=0.0)
    return corr / denom


def peak_ratio(corr_map: np.ndarray) -> tuple[float, float]:
    v = np.abs(corr_map).ravel()
    if v.size < 2:
        return np.inf, float(np.max(v)) if v.size else 0.0
    idx = int(np.argmax(v))
    peak = float(v[idx])
    v = v.copy()
    v[idx] = 0.0
    second = float(np.max(v))
    return peak / (second + 1e-12), peak


def detect_watermark_hierarchical(
    original_image: np.ndarray,
    received_image: np.ndarray,
    payload: dict,
    ratio_threshold: float = 1.05,
):
    levels = payload["levels"]
    wavelet = payload["wavelet"]
    Wsig = payload["watermark_signal"]

    co = dwt2_multilevel(original_image, levels=levels, wavelet=wavelet)
    cr = dwt2_multilevel(received_image, levels=levels, wavelet=wavelet)

    band_schedule = ["HH", "LH", "HL"]
    all_records = []

    for lev_idx in range(1, levels + 1):
        oH, oV, oD = co[lev_idx]
        rH, rV, rD = cr[lev_idx]
        band_map_o = {"LH": oH, "HL": oV, "HH": oD}
        band_map_r = {"LH": rH, "HL": rV, "HH": rD}

        selected = []
        for b in band_schedule:
            selected.append(b)
            corr_peaks = []
            for sb in selected:
                diff = band_map_r[sb] - band_map_o[sb]
                corr_map = normalized_2d_xcorr(diff, Wsig[(lev_idx, sb)])
                pr, pk = peak_ratio(corr_map)
                corr_peaks.append((pr, pk))

            ratio = float(np.mean([x[0] for x in corr_peaks]))
            peak = float(np.mean([x[1] for x in corr_peaks]))
            record = {
                "level": lev_idx,
                "bands": tuple(selected),
                "mean_peak_ratio": ratio,
                "mean_peak": peak,
                "detected": ratio >= ratio_threshold,
            }
            all_records.append(record)
            if record["detected"]:
                return True, record, all_records

    return False, None, all_records


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

    @property
    def name(self) -> str:
        return "dwt"

    def embed(self, image: Image.Image) -> Image.Image:
        img = image.convert("L")
        arr = img_as_float_or_uint8(np.asarray(img))
        x_hat, payload = embed_watermark_dwt(
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
        _ok, _rec, all_records = detect_watermark_hierarchical(
            o, c, payload, ratio_threshold=self.ratio_threshold
        )
        ratios = [r["mean_peak_ratio"] for r in all_records]
        max_ratio = max(ratios) if ratios else 0.0
        return float(np.clip((max_ratio - 1.0) / max(self.ratio_threshold - 1.0, 1e-6), 0.0, 1.0))


def img_as_float_or_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    return arr.astype(np.float64)
