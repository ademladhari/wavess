from __future__ import annotations

import os

import numpy as np
from PIL import Image

from wmbench.metrics.image import (
    compute_nmi_repeated,
    compute_psnr_repeated,
    compute_ssim_repeated,
)


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _pairs(originals: list[Image.Image], attacked: list[Image.Image]) -> tuple[list[Image.Image], list[Image.Image]]:
    if len(originals) != len(attacked):
        raise ValueError("originals and attacked must have same length")
    return originals, attacked


def compute_psnr(attacked: list[Image.Image], originals: list[Image.Image]) -> float:
    a, b = _pairs(originals, attacked)
    return _mean(compute_psnr_repeated(b, a, num_workers=os.cpu_count() or 4, verbose=False))


def compute_ssim(attacked: list[Image.Image], originals: list[Image.Image]) -> float:
    a, b = _pairs(originals, attacked)
    return _mean(compute_ssim_repeated(b, a, num_workers=os.cpu_count() or 4, verbose=False))


def compute_nmi(attacked: list[Image.Image], originals: list[Image.Image]) -> float:
    a, b = _pairs(originals, attacked)
    return _mean(compute_nmi_repeated(b, a, num_workers=os.cpu_count() or 4, verbose=False))
