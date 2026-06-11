from __future__ import annotations

import torch


def fft2_shifted(latents: torch.Tensor) -> torch.Tensor:
    """Return 2D FFT with centered frequencies."""
    # cuFFT half/complex-half requires power-of-2 sizes; fp32 FFT works for any latent shape.
    if latents.is_cuda and latents.dtype in (torch.float16, torch.bfloat16):
        x = latents.float()
        return torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    return torch.fft.fftshift(torch.fft.fft2(latents, dim=(-2, -1)), dim=(-2, -1))


def ifft2_shifted(freqs: torch.Tensor) -> torch.Tensor:
    """Return inverse FFT from centered frequencies."""
    return torch.fft.ifft2(torch.fft.ifftshift(freqs, dim=(-2, -1)), dim=(-2, -1))


def circular_mask(
    height: int,
    width: int,
    radius: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a centered circular mask in FFT space."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= float(radius**2)


def ring_bins(height: int, width: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return integer ring index per coefficient measured from FFT center."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radial_dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return radial_dist.floor().to(torch.long)
