"""Random binary watermark sampler (Sec. III-A, III-C)."""

from __future__ import annotations

from typing import Optional

import torch


def sample_watermarks(
    batch_size: int,
    n_bits: int,
    device: str | torch.device = "cpu",
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a tensor of shape (batch_size, n_bits) with i.i.d. Bernoulli(0.5) bits."""

    device = torch.device(device)
    if generator is None:
        w = torch.randint(0, 2, (batch_size, n_bits), device=device)
    else:
        w = torch.randint(0, 2, (batch_size, n_bits), device=device, generator=generator)
    return w.to(dtype=dtype)
