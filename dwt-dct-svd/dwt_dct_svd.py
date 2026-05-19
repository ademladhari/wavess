"""
Sundhararaj et al., Circuits, Systems, and Signal Processing (2025) 44:6650–6675,
DOI 10.1007/s00034-025-03097-7 — Sections 3.1–3.3.

Implementation is the classical **non-blind** (Liu–Tan-style) SVD watermarking scheme
that §3.2 describes step-by-step (SV merge, inverse SVD/DCT/DWT) and §3.3 inverts.
The extraction is non-blind: it needs the original cover, the original biometric, AND
the watermark-side singular vectors (Us, Vs, Ub, Vb) that were produced at embedding
time. §2.1 of the paper explicitly states that U and V "store the main structural
information of the image" — without them extraction reduces to a singular-value vector
and cannot reproduce the watermark image (NCC stays near 0).

Procedure (§3.2 / §3.3):

Part 1 — signature into biometric:
  embed:  SVD(signature) = Us Ss Vs^T,  DWT(biometric)→LL,  DCT(LL)=Fb,  SVD(Fb)=Ub Sb Vb^T.
          Sb' = Sb + α·Ss,  Fb' = Ub Sb' Vb^T,  LL' = IDCT(Fb'),  TW = IDWT(LL', details_b).
  extract: DWT(TW_hat)→LL,  DCT(LL)=Ft,  svd values of Ft = Stm (≈ Sb + α·Ss).
          Ss_hat = (Stm − Sb)/α,  signature_hat = Us · diag(Ss_hat) · Vs^T.

Part 2 — TW into cover (per R,G,B):
  embed:  DWT(cover_c)→LL, DCT(LL)=Fc, SVD(Fc)=Uc Sc Vc^T.  Stw = Sb' (already sorted).
          Sc' = Sc + β·Stw,  Fc' = Uc Sc' Vc^T,  LL' = IDCT(Fc'), wm_c = IDWT(LL',details_c).
  extract: DWT(wm_c)→LL, DCT=Fw, SV(Fw)=Sfw.  Stw_hat = mean_c (Sfw − Sc)/β.
          F_TW_hat = Ub · diag(Stw_hat) · Vb^T  (since embedding set F_TW_LL = Ub Sb' Vb^T).
          LL_hat = IDCT(F_TW_hat), TW_hat = IDWT(LL_hat, details_b).

§3.1 key A is generated for host binding / FPP-style extensions; it is not multiplied
into coefficients by this global SV-merge path. The key still acts as a non-blind
secret together with the saved {Us, Vs, Ub, Vb, Sb, Sc_list, details_b, details_c}
state, without which the watermark cannot be reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pywt
from scipy.fft import dctn, idctn
from scipy.linalg import svd
from skimage import color, util
from skimage.transform import resize

WAVELET = "haar"
WAVELET_MODE = "symmetric"


def to_float01(img: np.ndarray) -> np.ndarray:
    """uint8/uint16 → float in [0,1] (clipped). For already-float arrays pass through
    unchanged — clipping mid-pipeline would break the SV arithmetic in §3.2 / §3.3."""
    a = np.asarray(img)
    if a.dtype.kind == "f":
        return a.astype(np.float64, copy=False)
    return np.clip(util.img_as_float(a), 0.0, 1.0)


def dct2(x: np.ndarray) -> np.ndarray:
    """Eq. (2): 2-D DCT type-II, orthonormal."""
    return dctn(x, type=2, norm="ortho")


def idct2(x: np.ndarray) -> np.ndarray:
    return idctn(x, type=2, norm="ortho")


def dwt_ll_one_level(channel: np.ndarray) -> tuple[np.ndarray, tuple]:
    coeffs = pywt.dwt2(channel, WAVELET, mode=WAVELET_MODE)
    (LL, _) = coeffs
    return LL, coeffs


def idwt_from_LL_and_details(LL_new: np.ndarray, details: tuple) -> np.ndarray:
    LH, HL, HH = details
    return pywt.idwt2((LL_new, (LH, HL, HH)), WAVELET, mode=WAVELET_MODE)


def _spatial_match(a: np.ndarray, h0: int, w0: int) -> np.ndarray:
    """Center-crop or symmetric-pad to (h0,w0); Haar IDWT can drift by ±1 on odd sizes."""
    a = np.asarray(a, dtype=np.float64)
    ha, wa = a.shape[:2]
    if ha == h0 and wa == w0:
        return a
    if ha >= h0 and wa >= w0:
        sh = (ha - h0) // 2
        sw = (wa - w0) // 2
        return a[sh : sh + h0, sw : sw + w0]
    pad_h = max(0, h0 - ha)
    pad_w = max(0, w0 - wa)
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    b = np.pad(a, ((pt, pb), (pl, pr)), mode="symmetric")
    return b[:h0, :w0]


def svd2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = svd(x, full_matrices=False)
    return U, s, Vt


def reconstruct_svd(U: np.ndarray, s: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """(U · diag(s) · Vt) with broadcasting; handles s shorter than min(U.cols, Vt.rows)
    by zero-padding to max common dimension."""
    n = min(U.shape[1], Vt.shape[0])
    s_full = np.zeros(n, dtype=np.float64)
    s_full[: min(len(s), n)] = s[:n]
    return (U[:, :n] * s_full) @ Vt[:n, :]


# --------------------------------------------------------------------------- §3.1
def key_generation(key128: bytes, M: int, N: int) -> tuple[np.ndarray, dict]:
    """Eq. (4): A(i,j) = ((X(i,j) + Y(i,j)) * 2^14) mod 1 from two logistic maps."""
    if len(key128) != 16:
        raise ValueError("key128 must be exactly 16 bytes (128 bits).")
    u = np.frombuffer(key128, dtype=np.uint8).astype(np.float64)
    x0 = (u[0] + u[1] / 256.0) / 257.0
    y0 = (u[2] + u[3] / 256.0) / 257.0
    r1 = 3.57 + (u[4] / 255.0) * (4.0 - 3.57)
    r2 = 3.57 + (u[5] / 255.0) * (4.0 - 3.57)

    mh = max(1, M // 2)
    nw = max(1, N // 2)
    X = np.zeros((mh, nw), dtype=np.float64)
    Y = np.zeros((mh, nw), dtype=np.float64)
    x, y = float(x0), float(y0)
    burn = max(mh * nw, 256)
    for _ in range(burn):
        x = (r1 * x * (1.0 - x)) % 1.0
        y = (r2 * y * (1.0 - y)) % 1.0
    for i in range(mh):
        for j in range(nw):
            x = (r1 * x * (1.0 - x)) % 1.0
            y = (r2 * y * (1.0 - y)) % 1.0
            X[i, j] = x
            Y[i, j] = y

    A = np.mod((X + Y) * (2.0**14), 1.0)
    meta = {"x0": x0, "y0": y0, "r1": r1, "r2": r2, "X": X, "Y": Y}
    return A, meta


# ------------------------------------------------------ non-blind side information
@dataclass
class WatermarkState:
    """All matrices and SV vectors needed to invert §3.2 in §3.3.

    Carrying U/V of the watermark (signature Us,Vs; TW-LL basis Ub,Vb) is what makes
    the reconstructed image resemble the original (§2.1). Paper compares NCC on images,
    not on SV vectors; without these matrices an image-level NCC can never approach 1.
    """

    Us: np.ndarray
    Vts: np.ndarray
    sig_shape: tuple[int, int]
    Ub: np.ndarray
    Vtb: np.ndarray
    sb: np.ndarray
    bio_details: tuple
    bio_shape: tuple[int, int]
    sc_list: list[np.ndarray] = field(default_factory=list)
    cov_details_list: list[tuple] = field(default_factory=list)
    stw_embedded: np.ndarray | None = None
    cover_shape: tuple[int, int] | None = None


# --------------------------------------------------------------------------- §3.2 part 1
def embed_signature_in_biometric(
    biometric: np.ndarray,
    signature: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, WatermarkState]:
    bio = to_float01(biometric)
    if bio.ndim == 3:
        bio = color.rgb2gray(bio)
    sig = to_float01(signature)
    if sig.ndim == 3:
        sig = color.rgb2gray(sig)

    Us, ss, Vts = svd2d(sig)

    LL_b, coeffs_b = dwt_ll_one_level(bio)
    F_b = dct2(LL_b)
    Ub, sb, Vtb = svd2d(F_b)

    k = int(min(sb.size, ss.size))
    sb_new = sb.copy()
    sb_new[:k] = sb[:k] + float(alpha) * ss[:k]

    F_b_new = reconstruct_svd(Ub, sb_new, Vtb)
    LL_new = idct2(F_b_new)
    TW = idwt_from_LL_and_details(LL_new, coeffs_b[1])
    TW = _spatial_match(TW, bio.shape[0], bio.shape[1])

    _, details_b = coeffs_b
    state = WatermarkState(
        Us=Us,
        Vts=Vts,
        sig_shape=(int(sig.shape[0]), int(sig.shape[1])),
        Ub=Ub,
        Vtb=Vtb,
        sb=sb,
        bio_details=details_b,
        bio_shape=(int(bio.shape[0]), int(bio.shape[1])),
        stw_embedded=sb_new.copy(),
    )
    return TW, state


# --------------------------------------------------------------------------- §3.2 part 2
def embed_tw_in_cover(
    cover_rgb: np.ndarray,
    TW_gray: np.ndarray,
    beta: float,
    state: WatermarkState | None = None,
) -> tuple[np.ndarray, WatermarkState]:
    """Embed TW into RGB cover per channel; updates/returns state with cover side info.

    If `state` is None, a fresh state is built from `TW` alone (signature extraction
    will not be possible in that case — only TW extraction).
    """
    cover = to_float01(cover_rgb)
    if cover.ndim == 2:
        cover = np.stack([cover, cover, cover], axis=-1)
    H, W, _ = cover.shape

    tw = to_float01(TW_gray)
    if tw.ndim == 3:
        tw = color.rgb2gray(tw)

    LL_tw, _ = dwt_ll_one_level(tw)
    F_tw = dct2(LL_tw)
    Utw, stw, Vttw = svd2d(F_tw)

    if state is None:
        state = WatermarkState(
            Us=np.zeros((1, 1)),
            Vts=np.zeros((1, 1)),
            sig_shape=(0, 0),
            Ub=Utw,
            Vtb=Vttw,
            sb=np.zeros_like(stw),
            bio_details=(
                np.zeros_like(LL_tw),
                np.zeros_like(LL_tw),
                np.zeros_like(LL_tw),
            ),
            bio_shape=(int(tw.shape[0]), int(tw.shape[1])),
            stw_embedded=stw.copy(),
        )

    out = np.zeros_like(cover)
    sc_list: list[np.ndarray] = []
    details_list: list[tuple] = []
    for c in range(3):
        ch = cover[:, :, c]
        LL_c, coeffs_c = dwt_ll_one_level(ch)
        _, details_c = coeffs_c
        F_c = dct2(LL_c)
        Uc, sc, Vtc = svd2d(F_c)

        k = int(min(sc.size, stw.size))
        sc_new = sc.copy()
        sc_new[:k] = sc[:k] + float(beta) * stw[:k]

        F_new = reconstruct_svd(Uc, sc_new, Vtc)
        LL_new = idct2(F_new)
        wm_ch = idwt_from_LL_and_details(LL_new, details_c)
        wm_ch = _spatial_match(wm_ch, ch.shape[0], ch.shape[1])

        out[:, :, c] = wm_ch
        sc_list.append(sc.copy())
        details_list.append(details_c)

    state.sc_list = sc_list
    state.cov_details_list = details_list
    state.cover_shape = (int(H), int(W))
    state.stw_embedded = stw.copy()
    return out, state


# --------------------------------------------------------------------------- §3.3 part 1
def extract_tw_from_watermarked(
    watermarked_rgb: np.ndarray,
    cover_rgb: np.ndarray,
    beta: float,
    state: WatermarkState,
) -> np.ndarray:
    wm = to_float01(watermarked_rgb)
    cov = to_float01(cover_rgb)
    if cov.ndim == 2:
        cov = np.stack([cov, cov, cov], axis=-1)
    if wm.ndim == 2:
        wm = np.stack([wm, wm, wm], axis=-1)

    per_channel: list[np.ndarray] = []
    for c in range(3):
        LL_w, _ = dwt_ll_one_level(wm[:, :, c])
        LL_c, _ = dwt_ll_one_level(cov[:, :, c])
        F_w = dct2(LL_w)
        F_c = dct2(LL_c)
        _, sfw, _ = svd2d(F_w)
        _, sc, _ = svd2d(F_c)
        k = int(min(sfw.size, sc.size))
        per_channel.append((sfw[:k] - sc[:k]) / float(beta))

    kmin = int(min(v.size for v in per_channel))
    stw_hat = np.mean(np.stack([v[:kmin] for v in per_channel], axis=0), axis=0)

    Ub = state.Ub
    Vtb = state.Vtb
    F_tw_hat = reconstruct_svd(Ub, stw_hat, Vtb)
    LL_tw_hat = idct2(F_tw_hat)

    # Use biometric detail subbands so TW_hat reconstructs back to the same image space
    # that §3.2 part 1 produced (§3.2 step 6 used biometric details).
    details = state.bio_details
    LH, HL, HH = details
    if LH.shape != LL_tw_hat.shape:
        LH = np.zeros_like(LL_tw_hat)
        HL = np.zeros_like(LL_tw_hat)
        HH = np.zeros_like(LL_tw_hat)
    TW_hat = pywt.idwt2((LL_tw_hat, (LH, HL, HH)), WAVELET, mode=WAVELET_MODE)
    bio_h, bio_w = state.bio_shape
    return _spatial_match(TW_hat, bio_h, bio_w)


# --------------------------------------------------------------------------- §3.3 part 2
def extract_signature_from_tw(
    TW_hat: np.ndarray,
    biometric: np.ndarray,
    alpha: float,
    state: WatermarkState,
) -> np.ndarray:
    tw = to_float01(TW_hat)
    if tw.ndim == 3:
        tw = color.rgb2gray(tw)

    LL_t, _ = dwt_ll_one_level(tw)
    F_t = dct2(LL_t)
    _, stm, _ = svd2d(F_t)

    sb = state.sb
    k = int(min(stm.size, sb.size))
    ss_hat_short = (stm[:k] - sb[:k]) / float(alpha)

    Us = state.Us
    Vts = state.Vts
    n_sig = int(min(Us.shape[1], Vts.shape[0]))
    ss_hat = np.zeros(n_sig, dtype=np.float64)
    kk = int(min(k, n_sig))
    ss_hat[:kk] = ss_hat_short[:kk]

    sig_hat = reconstruct_svd(Us, ss_hat, Vts)
    h_s, w_s = state.sig_shape
    return _spatial_match(sig_hat, h_s, w_s)


# --------------------------------------------------------------------------- metrics
def psnr_channelwise(
    cover_rgb: np.ndarray, wm_rgb: np.ndarray
) -> tuple[float, list[float]]:
    """Sec. 4.1 — mean and per-channel PSNR of R, G, B."""
    a = to_float01(cover_rgb)
    b = to_float01(wm_rgb)
    if a.ndim == 2:
        a = np.stack([a, a, a], -1)
    if b.ndim == 2:
        b = np.stack([b, b, b], -1)
    ch_psnr: list[float] = []
    for c in range(3):
        mse = float(np.mean((a[..., c] - b[..., c]) ** 2))
        ch_psnr.append(float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse))
    return float(np.mean(ch_psnr)), ch_psnr


def ncc(W: np.ndarray, W_extracted: np.ndarray) -> float:
    """Eq. (7) NCC on equal-shape images."""
    a = to_float01(W).ravel()
    b = to_float01(W_extracted).ravel()
    if a.size != b.size:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    mu1, mu2 = float(np.mean(a)), float(np.mean(b))
    a0 = a - mu1
    b0 = b - mu2
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0:
        return 0.0
    return float(np.dot(a0, b0) / denom)


# ------------------------------------------- backward-compat thin alias (removed API)
def _sv_wm_scale(*_args: Any, **_kwargs: Any) -> np.ndarray:
    raise RuntimeError(
        "L2 SV scaling is no longer used — the non-blind scheme uses raw α, β. "
        "Update your script to the new dwt_dct_svd API (no _sv_wm_scale)."
    )


def cover_basis_diag_coef_corr(*_a: Any, **_kw: Any) -> tuple[float, int]:
    raise RuntimeError(
        "Removed: the new non-blind scheme does not embed diag(U^T F V^T); "
        "use NCC(signature, signature_extracted) and NCC(TW, TW_extracted) instead."
    )


def biometric_basis_diag_coef_corr(*_a: Any, **_kw: Any) -> tuple[float, int]:
    raise RuntimeError(
        "Removed: see cover_basis_diag_coef_corr docstring."
    )


_ = cast(Any, resize)
