"""
Shared attack suite for per-method run_benchmark.py scripts.

Fixed strengths (moderate / wmbench relative strength 0.5 where noted):
  rotation          22.5 deg
  resized_crop      scale 0.75 (RandomResizedCrop)
  erasing           12.5% area
  brightness        factor 1.5
  contrast          factor 1.5
  blur              Gaussian radius 4 px
  resize_90         scale 0.9 then back
  jpeg_q50          quality 50
  crop              center 85%
  gaussian          sigma 25 (0-255) / 25/255 ([0,1])
  combo_geometric   rotation 22.5 -> resized_crop 0.75
  combo_photometric brightness 1.5 -> contrast 1.5
  combined          crop 85% -> JPEG Q50 -> gaussian sigma 15
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AttackSpec:
    name: str
    description: str


ATTACKS: tuple[AttackSpec, ...] = (
    AttackSpec("identity", "no distortion"),
    AttackSpec("rotation", "rotate 22.5 deg (wmbench rel=0.5)"),
    AttackSpec("resized_crop", "random resized crop scale 0.75, square"),
    AttackSpec("erasing", "random erasing 12.5% area"),
    AttackSpec("brightness", "brightness factor 1.5"),
    AttackSpec("contrast", "contrast factor 1.5"),
    AttackSpec("blur", "Gaussian blur radius 4 px"),
    AttackSpec("resize_90", "resize to 90% then back"),
    AttackSpec("jpeg_q50", "JPEG quality 50"),
    AttackSpec("crop", "center crop 85% then resize back"),
    AttackSpec("gaussian", "additive Gaussian noise sigma=25 (0-255 scale)"),
    AttackSpec("combo_geometric", "rotation 22.5 -> resized_crop 0.75"),
    AttackSpec("combo_photometric", "brightness 1.5 -> contrast 1.5"),
    AttackSpec("combined", "crop 85% -> JPEG Q50 -> Gaussian sigma=15"),
)

# Fixed absolute strengths
ROTATION_DEG = 22.5
RESIZED_CROP_SCALE = 0.75
ERASING_SCALE = 0.125
BRIGHTNESS_FACTOR = 1.5
CONTRAST_FACTOR = 1.5
BLUR_RADIUS = 4.0
RESIZE_FRAC = 0.9
CROP_FRAC = 0.85
JPEG_QUALITY = 50
GAUSSIAN_SIGMA_255 = 25.0
COMBINED_GAUSSIAN_SIGMA = 15.0


def _as_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB")


def _jpeg(im: Image.Image, quality: int) -> Image.Image:
    buf = BytesIO()
    _as_rgb(im).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _center_crop_resize(im: Image.Image, frac: float) -> Image.Image:
    im = _as_rgb(im)
    w, h = im.size
    nw, nh = max(1, int(w * frac)), max(1, int(h * frac))
    left, top = (w - nw) // 2, (h - nh) // 2
    out = im.crop((left, top, left + nw, top + nh))
    return out.resize((w, h), Image.Resampling.LANCZOS)


def _resize_frac(im: Image.Image, frac: float) -> Image.Image:
    im = _as_rgb(im)
    w, h = im.size
    nw, nh = max(1, int(w * frac)), max(1, int(h * frac))
    out = im.resize((nw, nh), Image.Resampling.LANCZOS)
    return out.resize((w, h), Image.Resampling.LANCZOS)


def _gaussian_noise_pil(im: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    arr = np.asarray(_as_rgb(im), dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _rotate(im: Image.Image, degrees: float) -> Image.Image:
    return TF.rotate(_as_rgb(im), float(degrees))


def _seed_all(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))


def _resized_crop(im: Image.Image, scale: float, seed: int) -> Image.Image:
    im = _as_rgb(im)
    _seed_all(seed)
    i, j, h, w = T.RandomResizedCrop.get_params(im, scale=(scale, scale), ratio=(1.0, 1.0))
    return TF.resized_crop(im, i, j, h, w, im.size)


def _erasing(im: Image.Image, scale: float, seed: int) -> Image.Image:
    im = _as_rgb(im)
    t = TF.to_tensor(im).unsqueeze(0)
    _seed_all(seed)
    i, j, h, w, v = T.RandomErasing.get_params(t, scale=(scale, scale), ratio=(1.0, 1.0), value=[0.0])
    out = TF.erase(t, int(i), int(j), int(h), int(w), v)
    return TF.to_pil_image(out.squeeze(0).clamp(0.0, 1.0))


def apply_attack_rgb(name: str, im: Image.Image, *, seed: int = 0) -> Image.Image:
    if name == "identity":
        return im.copy()
    if name == "rotation":
        return _rotate(im, ROTATION_DEG)
    if name == "resized_crop":
        return _resized_crop(im, RESIZED_CROP_SCALE, seed)
    if name == "erasing":
        return _erasing(im, ERASING_SCALE, seed)
    if name == "brightness":
        return ImageEnhance.Brightness(_as_rgb(im)).enhance(BRIGHTNESS_FACTOR)
    if name == "contrast":
        return ImageEnhance.Contrast(_as_rgb(im)).enhance(CONTRAST_FACTOR)
    if name == "blur":
        im = _as_rgb(im)
        return im.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)) if BLUR_RADIUS > 0 else im.copy()
    if name == "resize_90":
        return _resize_frac(im, RESIZE_FRAC)
    if name == "jpeg_q50":
        return _jpeg(im, JPEG_QUALITY)
    if name == "crop":
        return _center_crop_resize(im, CROP_FRAC)
    if name == "gaussian":
        return _gaussian_noise_pil(im, GAUSSIAN_SIGMA_255, seed)
    if name == "combo_geometric":
        out = _rotate(im, ROTATION_DEG)
        return _resized_crop(out, RESIZED_CROP_SCALE, seed + 1)
    if name == "combo_photometric":
        out = ImageEnhance.Brightness(_as_rgb(im)).enhance(BRIGHTNESS_FACTOR)
        return ImageEnhance.Contrast(out).enhance(CONTRAST_FACTOR)
    if name == "combined":
        out = _center_crop_resize(im, CROP_FRAC)
        out = _jpeg(out, JPEG_QUALITY)
        return _gaussian_noise_pil(out, COMBINED_GAUSSIAN_SIGMA, seed + 1)
    raise ValueError(f"Unknown attack {name!r}")


def apply_attack_gray(name: str, img: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Grayscale float64 in [0, 255]. Identity keeps float; other attacks use 8-bit PIL."""
    if name == "identity":
        return np.clip(img.copy(), 0.0, 255.0)
    pil = Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8), mode="L")
    out = apply_attack_rgb(name, pil, seed=seed)
    return np.asarray(out.convert("L"), dtype=np.float64)


def apply_attack_tensor(name: str, img: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """img: float (3, H, W) in [0, 1]."""
    arr = (img.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    pil = Image.fromarray(arr, mode="RGB")
    out = apply_attack_rgb(name, pil, seed=seed)
    arr2 = np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr2, (2, 0, 1))).to(img.device)
