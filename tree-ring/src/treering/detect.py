from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import ncx2

from .fourier import fft2_shifted
from .keygen import KeyMaterial


@dataclass(slots=True)
class DetectionResult:
    distance: float
    p_value: float
    detected_by_distance: bool
    detected_by_pvalue: bool


def _masked_complex_l1(freqs: torch.Tensor, key_fft: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = torch.abs(freqs[:, :, mask] - key_fft[:, mask])
    return values.mean(dim=(1, 2))


def _estimate_sigma2(
    freqs: torch.Tensor,
    mask: torch.Tensor,
    method: str,
) -> torch.Tensor:
    if method == "mask":
        region = freqs[:, :, mask]
    elif method == "outside_mask":
        region = freqs[:, :, ~mask]
    elif method == "global":
        region = freqs.reshape(freqs.shape[0], freqs.shape[1], -1)
    else:
        raise ValueError(f"Unsupported variance estimation method: {method}")
    return torch.mean(torch.abs(region) ** 2, dim=(1, 2)).clamp_min(1e-8)


def _masked_eta(
    freqs: torch.Tensor,
    key_fft: torch.Tensor,
    mask: torch.Tensor,
    variance_estimation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = freqs[:, :, mask]
    k = key_fft[:, mask]
    sigma2 = _estimate_sigma2(freqs, mask, method=variance_estimation)
    # Complex normal mapped to real-valued ncx2 uses factor 2.
    eta = 2.0 * torch.sum(torch.abs(k.unsqueeze(0) - y) ** 2, dim=(1, 2)) / sigma2
    return eta, sigma2


def detect_watermark(
    inverted_latents: torch.Tensor,
    key: KeyMaterial,
    threshold: float,
    alpha: float,
    variance_estimation: str = "outside_mask",
    pvalue_tail: str = "lower",
) -> list[DetectionResult]:
    """Run Eq.(3) L1 detection and Eq.(5)/(6) p-value test."""
    freqs = fft2_shifted(inverted_latents)
    key_fft = key.key_fft.to(device=inverted_latents.device, dtype=freqs.dtype)
    mask = key.mask.to(device=inverted_latents.device)

    l1_scores = _masked_complex_l1(freqs, key_fft, mask)
    eta, sigma2 = _masked_eta(freqs, key_fft, mask, variance_estimation=variance_estimation)

    dof = int(2 * mask.sum().item() * key_fft.shape[0])
    key_norm = torch.sum(torch.abs(key_fft[:, mask]) ** 2)

    results: list[DetectionResult] = []
    for idx in range(inverted_latents.shape[0]):
        sigma2_i = float(sigma2[idx].item())
        lambda_i = float((2.0 * key_norm / sigma2_i).item())
        eta_i = float(eta[idx].item())
        cdf = float(ncx2.cdf(eta_i, dof, lambda_i))
        if pvalue_tail == "lower":
            p_value = cdf
        elif pvalue_tail == "upper":
            p_value = 1.0 - cdf
        else:
            raise ValueError(f"Unsupported p-value tail: {pvalue_tail}")
        distance = float(l1_scores[idx].item())
        results.append(
            DetectionResult(
                distance=distance,
                p_value=p_value,
                detected_by_distance=distance < threshold,
                detected_by_pvalue=p_value < alpha,
            )
        )
    return results
