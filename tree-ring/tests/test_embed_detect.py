from __future__ import annotations

import torch

from treering.detect import detect_watermark
from treering.embed import build_watermarked_latents
from treering.fourier import circular_mask
from treering.keygen import generate_key_material


def test_embed_and_detect_smoke() -> None:
    device = "cpu"
    channels, height, width = 4, 16, 16
    mask = circular_mask(height, width, radius=4, device=device)
    key = generate_key_material(
        channels=channels,
        height=height,
        width=width,
        mask=mask,
        variant="rand",
        seed=7,
        device=device,
    )
    latents = build_watermarked_latents(
        batch_size=2,
        channels=channels,
        height=height,
        width=width,
        key=key,
        device=device,
        dtype=torch.float32,
        seed=13,
    )
    results = detect_watermark(
        inverted_latents=latents,
        key=key,
        threshold=0.5,
        alpha=0.5,
    )
    assert len(results) == 2
    for res in results:
        assert res.distance >= 0
        assert 0.0 <= res.p_value <= 1.0
