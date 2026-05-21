"""The eight traditional image-processing attacks (Sec. IV-A.2 / Fig. 4).

All attack functions take and return a float tensor in [0, 1] of shape
(3, H, W). Batched inputs of shape (B, 3, H, W) are also supported.

Fixed-strength settings (from the paper):
  brightness factor = 3
  contrast factor   = 2
  Gaussian noise sigma = 0.05
  JPEG quality = 50
  median kernel = 5
  Gaussian blur kernel = 3
  resize factor = 0.9
  BM3D sigma_psd = 10
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


def _as_batch(img: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if img.dim() == 3:
        return img.unsqueeze(0), True
    return img, False


def _debatch(img: torch.Tensor, was_single: bool) -> torch.Tensor:
    return img[0] if was_single else img


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def brightness(img: torch.Tensor, factor: float = 3.0) -> torch.Tensor:
    batch, single = _as_batch(img)
    out = []
    for x in batch:
        pil = _tensor_to_pil(x)
        pil = ImageEnhance.Brightness(pil).enhance(float(factor))
        out.append(_pil_to_tensor(pil))
    y = torch.stack(out, dim=0).to(img.device)
    return _debatch(y, single)


def contrast(img: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    batch, single = _as_batch(img)
    out = []
    for x in batch:
        pil = _tensor_to_pil(x)
        pil = ImageEnhance.Contrast(pil).enhance(float(factor))
        out.append(_pil_to_tensor(pil))
    y = torch.stack(out, dim=0).to(img.device)
    return _debatch(y, single)


def gaussian_noise(img: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    noise = torch.randn_like(img) * float(sigma)
    return (img + noise).clamp(0.0, 1.0)


def jpeg_compression(img: torch.Tensor, quality: int = 50) -> torch.Tensor:
    batch, single = _as_batch(img)
    out = []
    for x in batch:
        pil = _tensor_to_pil(x)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        pil2 = Image.open(buf).convert("RGB")
        out.append(_pil_to_tensor(pil2))
    y = torch.stack(out, dim=0).to(img.device)
    return _debatch(y, single)


def _median_filter(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (B, C, H, W)
    pad = k // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    b, c, h, w = x.shape
    patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)  # (B, C, H, W, k, k)
    patches = patches.contiguous().view(b, c, h, w, k * k)
    return patches.median(dim=-1).values


def median_filter(img: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    batch, single = _as_batch(img)
    y = _median_filter(batch, int(kernel))
    return _debatch(y.clamp(0.0, 1.0), single)


def gaussian_blur(img: torch.Tensor, kernel: int = 3, sigma: float | None = None) -> torch.Tensor:
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    if sigma is None:
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g1d = torch.exp(-(ax**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    g2d = torch.outer(g1d, g1d).to(img.device)
    batch, single = _as_batch(img)
    c = batch.shape[1]
    weight = g2d.view(1, 1, k, k).repeat(c, 1, 1, 1)
    y = F.conv2d(batch, weight, padding=k // 2, groups=c)
    return _debatch(y.clamp(0.0, 1.0), single)


def resize_attack(img: torch.Tensor, factor: float = 0.9) -> torch.Tensor:
    batch, single = _as_batch(img)
    b, c, h, w = batch.shape
    nh = max(1, int(round(h * float(factor))))
    nw = max(1, int(round(w * float(factor))))
    small = F.interpolate(batch, size=(nh, nw), mode="bilinear", align_corners=False)
    y = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return _debatch(y.clamp(0.0, 1.0), single)


def bm3d_denoise(img: torch.Tensor, sigma_psd: float = 10.0) -> torch.Tensor:
    """BM3D denoising attack (Sec. IV-A.2). Expects sigma_psd on the 0..255 scale."""

    import bm3d  # lazy import; heavy dep

    batch, single = _as_batch(img)
    out = []
    for x in batch:
        arr = x.clamp(0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0)  # HWC in [0,1]
        arr = (arr * 255.0).astype(np.float32)
        denoised = bm3d.bm3d(arr, sigma_psd=float(sigma_psd))
        denoised = np.clip(denoised, 0.0, 255.0) / 255.0
        out.append(torch.from_numpy(denoised.astype(np.float32).transpose(2, 0, 1)))
    y = torch.stack(out, dim=0).to(img.device)
    return _debatch(y, single)


FIXED_ATTACK_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "brightness": lambda x: brightness(x, 3.0),
    "contrast": lambda x: contrast(x, 2.0),
    "gauss_noise": lambda x: gaussian_noise(x, 0.05),
    "jpeg": lambda x: jpeg_compression(x, 50),
    "median": lambda x: median_filter(x, 5),
    "gauss_blur": lambda x: gaussian_blur(x, 3),
    "resize": lambda x: resize_attack(x, 0.9),
    "bm3d": lambda x: bm3d_denoise(x, 10.0),
}


def apply_fixed(img: torch.Tensor, name: str) -> torch.Tensor:
    return FIXED_ATTACK_REGISTRY[name](img)


def apply_swept(img: torch.Tensor, name: str, strength) -> torch.Tensor:
    if name == "brightness":
        return brightness(img, float(strength))
    if name == "contrast":
        return contrast(img, float(strength))
    if name == "gauss_noise":
        return gaussian_noise(img, float(strength))
    if name == "jpeg":
        return jpeg_compression(img, int(strength))
    if name == "median":
        return median_filter(img, int(strength))
    if name == "gauss_blur":
        return gaussian_blur(img, int(strength))
    if name == "resize":
        return resize_attack(img, float(strength))
    if name == "bm3d":
        return bm3d_denoise(img, float(strength))
    raise KeyError(name)
