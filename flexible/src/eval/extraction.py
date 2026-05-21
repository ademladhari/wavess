"""Extraction-quality metrics (Sec. IV-A.3 / III-D).

* BitAcc: fraction of correctly decoded bits.
* TPR@0.01FPR: true-positive rate at a bit-count threshold tau such that the
  theoretical FPR under i.i.d. Bernoulli(0.5) matches is below 0.01.
"""

from __future__ import annotations

from math import comb

import torch


def bit_accuracy(wd: torch.Tensor, w: torch.Tensor) -> float:
    wd_bits = (wd > 0.5).to(torch.int64)
    w_bits = (w > 0.5).to(torch.int64)
    return (wd_bits == w_bits).float().mean().item()


def per_sample_matches(wd: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wd_bits = (wd > 0.5).to(torch.int64)
    w_bits = (w > 0.5).to(torch.int64)
    return (wd_bits == w_bits).sum(dim=-1)


def fpr_threshold(n_bits: int, fpr: float = 0.01) -> int:
    """Smallest k s.t. P(Bin(n, 0.5) >= k) <= fpr."""

    cum = 0.0
    denom = 2 ** n_bits
    # iterate from k = n_bits down, accumulating tail probability.
    for k in range(n_bits, -1, -1):
        cum += comb(n_bits, k) / denom
        if cum > fpr:
            return k + 1
    return 0


def tpr_at_fpr(wd: torch.Tensor, w: torch.Tensor, fpr: float = 0.01) -> tuple[float, int]:
    """Return (TPR, tau). TPR is the fraction of samples with >= tau matching bits."""

    n_bits = wd.shape[-1]
    tau = fpr_threshold(n_bits, fpr=fpr)
    matches = per_sample_matches(wd, w)
    tpr = (matches >= tau).float().mean().item()
    return tpr, tau
