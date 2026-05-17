from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct


def dct2(x: np.ndarray) -> np.ndarray:
    return dct(dct(x.T, norm="ortho").T, norm="ortho")


def idct2(x: np.ndarray) -> np.ndarray:
    return idct(idct(x.T, norm="ortho").T, norm="ortho")


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def top_magnitude_indices(coeffs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.abs(coeffs).ravel().copy()
    if flat.size <= 1:
        raise ValueError("DCT coefficient map is too small to select watermark indices")
    flat[0] = -np.inf  # Exclude the DC term.
    n = min(n, flat.size - 1)
    idx = np.argpartition(flat, -n)[-n:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return np.unravel_index(idx, coeffs.shape)


def generate_ground_truth_bits(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(length) > 0.5).astype(bool)


def bits_to_sign(bits: np.ndarray) -> np.ndarray:
    return np.where(bits.astype(bool), 1.0, -1.0).astype(np.float64)


def embed_bits(image: np.ndarray, bits: np.ndarray, alpha: float) -> np.ndarray:
    coeffs = dct2(image)
    rows, cols = top_magnitude_indices(coeffs, len(bits))
    signs = bits_to_sign(bits)
    coeffs_marked = coeffs.copy()
    coeffs_marked[rows, cols] = coeffs[rows, cols] * (1.0 + alpha * signs)
    return to_uint8(idct2(coeffs_marked)).astype(np.float64)


def extract_bits(
    original_image: np.ndarray,
    candidate_image: np.ndarray,
    length: int,
    alpha: float,
    eps: float = 1e-10,
) -> np.ndarray:
    c0 = dct2(original_image)
    c1 = dct2(candidate_image)
    rows, cols = top_magnitude_indices(c0, length)
    denom = np.where(np.abs(c0[rows, cols]) < eps, eps, c0[rows, cols])
    extracted = (c1[rows, cols] / denom - 1.0) / alpha
    return (extracted > 0).astype(bool)
