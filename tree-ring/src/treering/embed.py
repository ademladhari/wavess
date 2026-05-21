from __future__ import annotations

import torch

from .fourier import fft2_shifted, ifft2_shifted
from .keygen import KeyMaterial


def _enforce_hermitian(freqs: torch.Tensor) -> torch.Tensor:
    """Enforce F(-w)=conj(F(w)) so inverse FFT is real-valued."""
    height, width = freqs.shape[-2:]
    for y in range(height):
        yp = (-y) % height
        for x in range(width):
            xp = (-x) % width
            if (y > yp) or (y == yp and x > xp):
                continue
            a = freqs[..., y, x]
            b = freqs[..., yp, xp]
            if y == yp and x == xp:
                # Self-conjugate bins must be purely real.
                real_part = ((a + torch.conj(b)) * 0.5).real
                freqs[..., y, x] = real_part + 0j
            else:
                avg = (a + torch.conj(b)) * 0.5
                freqs[..., y, x] = avg
                freqs[..., yp, xp] = torch.conj(avg)
    return freqs


def apply_key_to_latents(latents: torch.Tensor, key: KeyMaterial) -> torch.Tensor:
    """Inject key coefficients in FFT space at masked positions."""
    if latents.ndim != 4:
        raise ValueError("Latents must have shape [B, C, H, W].")

    batch, channels, height, width = latents.shape
    if key.key_fft.shape != (channels, height, width):
        raise ValueError(
            f"Key shape {tuple(key.key_fft.shape)} does not match latent channels/spatial shape."
        )

    # Keep pipeline fast in fp16, but do Tree-Ring FFT math in fp32 for stability.
    fft_input = latents if latents.dtype == torch.float32 else latents.float()
    freqs = fft2_shifted(fft_input)
    key_fft = key.key_fft.to(device=latents.device, dtype=freqs.dtype)
    mask = key.mask.to(device=latents.device)

    for b_idx in range(batch):
        for c_idx in range(channels):
            freqs[b_idx, c_idx, mask] = key_fft[c_idx, mask]
    freqs = _enforce_hermitian(freqs)
    embedded = ifft2_shifted(freqs).real
    return embedded.to(dtype=latents.dtype)


def build_watermarked_latents(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    key: KeyMaterial,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
    latents = torch.randn(
        (batch_size, channels, height, width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return apply_key_to_latents(latents, key)
