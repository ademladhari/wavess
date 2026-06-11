"""Fixed attack suite for the simple benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class AttackSpec:
    name: str
    description: str


ATTACKS: tuple[AttackSpec, ...] = (
    AttackSpec("identity", "no distortion"),
    AttackSpec("jpeg_q50", "JPEG quality 50"),
    AttackSpec("crop", "center crop 85% then resize back"),
    AttackSpec("gaussian", "additive Gaussian noise sigma=25"),
    AttackSpec("regeneration", "Regen-VAE CompressAI quality=4"),
    AttackSpec("combined", "crop 85% -> JPEG Q50 -> Gaussian noise sigma=15"),
)


def apply_jpeg_q50(im: Image.Image) -> Image.Image:
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_crop(im: Image.Image, frac: float = 0.85) -> Image.Image:
    w, h = im.size
    nw, nh = max(1, int(w * frac)), max(1, int(h * frac))
    left, top = (w - nw) // 2, (h - nh) // 2
    out = im.crop((left, top, left + nw, top + nh))
    return out.resize((w, h), Image.Resampling.LANCZOS)


def apply_gaussian_noise(im: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


class AttackRunner:
    def __init__(self, device: torch.device, *, skip_regen: bool = False):
        self.device = device
        self.skip_regen = skip_regen
        self._regen_vae = None

    def _get_regen_vae(self):
        if self._regen_vae is None:
            from wmbench_regen_attacks import VAERegenAttack

            self._regen_vae = VAERegenAttack(device=self.device)
        return self._regen_vae

    def apply(self, name: str, im: Image.Image) -> Image.Image:
        if name == "identity":
            return im.copy()
        if name == "jpeg_q50":
            return apply_jpeg_q50(im)
        if name == "crop":
            return apply_crop(im, frac=0.85)
        if name == "gaussian":
            return apply_gaussian_noise(im, sigma=25.0)
        if name == "regeneration":
            if self.skip_regen:
                return im.copy()
            return self._get_regen_vae().apply(im, strength=4)
        if name == "combined":
            out = apply_crop(im, frac=0.85)
            out = apply_jpeg_q50(out)
            return apply_gaussian_noise(out, sigma=15.0, seed=1)
        raise ValueError(f"Unknown attack {name!r}")
