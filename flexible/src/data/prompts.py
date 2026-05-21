"""Prompt loader for Gustavosta/Stable-Diffusion-Prompts (Sec. IV-A.1)."""

from __future__ import annotations

import random
from typing import Optional

from datasets import load_dataset


def load_prompts(
    name: str = "Gustavosta/Stable-Diffusion-Prompts",
    split: str = "train",
    n: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> list[str]:
    """Return ``n`` prompts drawn deterministically given ``seed``.

    Passing ``n=None`` returns the full split.
    """

    ds = load_dataset(name, split=split, cache_dir=cache_dir)

    # Gustavosta/Stable-Diffusion-Prompts uses "Prompt"; fall back defensively.
    key = None
    for candidate in ("Prompt", "prompt", "text"):
        if candidate in ds.column_names:
            key = candidate
            break
    if key is None:
        raise ValueError(
            f"Cannot find a prompt column in dataset '{name}'. Columns: {ds.column_names}"
        )

    prompts = [row[key] for row in ds]
    prompts = [p for p in prompts if isinstance(p, str) and len(p.strip()) > 0]

    if n is None:
        return prompts

    rng = random.Random(seed)
    if n >= len(prompts):
        out = list(prompts)
        rng.shuffle(out)
        return out
    return rng.sample(prompts, n)
