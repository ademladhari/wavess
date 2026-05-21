from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .fourier import ring_bins

KeyVariant = Literal["zeros", "rand", "rings"]


@dataclass(slots=True)
class KeyMaterial:
    key_fft: torch.Tensor  # Complex tensor [C, H, W]
    mask: torch.Tensor  # Bool tensor [H, W]
    variant: KeyVariant
    seed: int


def _conjugate_index(y: int, x: int, height: int, width: int) -> tuple[int, int]:
    # Coordinates are in fftshifted layout; conjugate partner is (-f_y, -f_x).
    return ((-y) % height, (-x) % width)


def _rand_complex(
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    # Scale by sqrt(2) so E[|z|^2] ~= 1 for standard complex normal.
    real = torch.randn(shape, generator=generator, device=device)
    imag = torch.randn(shape, generator=generator, device=device)
    return (real + 1j * imag) / (2.0**0.5)


def generate_key_material(
    channels: int,
    height: int,
    width: int,
    mask: torch.Tensor,
    variant: KeyVariant,
    seed: int,
    device: torch.device | str = "cpu",
) -> KeyMaterial:
    if variant not in {"zeros", "rand", "rings"}:
        raise ValueError(f"Unsupported key variant: {variant}")

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    key_fft = torch.zeros((channels, height, width), dtype=torch.complex64, device=torch_device)

    if variant == "zeros":
        return KeyMaterial(key_fft=key_fft, mask=mask, variant=variant, seed=seed)

    if variant == "rand":
        coords = torch.nonzero(mask, as_tuple=False)
        for yx in coords:
            y = int(yx[0].item())
            x = int(yx[1].item())
            yp, xp = _conjugate_index(y, x, height, width)
            if (y > yp) or (y == yp and x > xp):
                continue
            sample = _rand_complex((channels,), generator, torch_device)
            if y == yp and x == xp:
                sample = sample.real + 0j
            key_fft[:, y, x] = sample
            key_fft[:, yp, xp] = torch.conj(sample)
        return KeyMaterial(key_fft=key_fft, mask=mask, variant=variant, seed=seed)

    bins = ring_bins(height, width, device=torch_device)
    ring_ids = torch.unique(bins[mask]).tolist()
    for channel in range(channels):
        for ring_id in ring_ids:
            ring_region = mask & (bins == int(ring_id))
            if not torch.any(ring_region):
                continue
            coords = torch.nonzero(ring_region, as_tuple=False)
            ring_value = _rand_complex((1,), generator, torch_device).squeeze(0)
            for yx in coords:
                y = int(yx[0].item())
                x = int(yx[1].item())
                yp, xp = _conjugate_index(y, x, height, width)
                if not bool(ring_region[yp, xp].item()):
                    continue
                if (y > yp) or (y == yp and x > xp):
                    continue
                sample = ring_value
                if y == yp and x == xp:
                    sample = sample.real + 0j
                key_fft[channel, y, x] = sample
                key_fft[channel, yp, xp] = torch.conj(sample)
    return KeyMaterial(key_fft=key_fft, mask=mask, variant=variant, seed=seed)
