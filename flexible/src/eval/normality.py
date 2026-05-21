"""Normality tests for Table II (Sec. IV-C).

Given a set of watermarked latent vectors produced by E with a fixed 48-bit
watermark, evaluate how close each latent dimension is to N(0, 1) using:

  * D'Agostino K-squared test (skew + kurtosis)
  * Kolmogorov-Smirnov test against N(0, 1)
  * Jarque-Bera test (skew + kurtosis)
  * Wasserstein distance between empirical distribution and N(0, 1)

Reports global mean, global std, and the fraction of latent dimensions that
pass each test at alpha=0.05.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import jarque_bera, kstest, normaltest, wasserstein_distance


@dataclass
class NormalityResult:
    n_samples: int
    global_mean: float
    global_std: float
    dagostino_pass_rate: float
    ks_pass_rate: float
    jarque_bera_pass_rate: float
    mean_wasserstein: float


def evaluate_normality(latents: np.ndarray, alpha: float = 0.05) -> NormalityResult:
    """Evaluate normality of per-dim distributions.

    ``latents`` has shape (N, ...) where every sample contains the same latent
    layout; it is flattened to (N, D) and each of the D dims is tested
    independently.
    """

    n = latents.shape[0]
    flat = latents.reshape(n, -1).astype(np.float64)
    d = flat.shape[1]

    global_mean = float(flat.mean())
    global_std = float(flat.std())

    dagostino_pass = 0
    ks_pass = 0
    jb_pass = 0
    wasserstein_vals = []

    # Reference Gaussian for Wasserstein.
    ref_sample = np.random.default_rng(0).standard_normal(max(2000, n))

    for dim in range(d):
        col = flat[:, dim]
        try:
            _, p_da = normaltest(col)
            if p_da > alpha:
                dagostino_pass += 1
        except Exception:
            pass
        try:
            _, p_ks = kstest(col, "norm")
            if p_ks > alpha:
                ks_pass += 1
        except Exception:
            pass
        try:
            _, p_jb = jarque_bera(col)
            if p_jb > alpha:
                jb_pass += 1
        except Exception:
            pass

        try:
            wasserstein_vals.append(float(wasserstein_distance(col, ref_sample)))
        except Exception:
            pass

    return NormalityResult(
        n_samples=n,
        global_mean=global_mean,
        global_std=global_std,
        dagostino_pass_rate=dagostino_pass / max(d, 1),
        ks_pass_rate=ks_pass / max(d, 1),
        jarque_bera_pass_rate=jb_pass / max(d, 1),
        mean_wasserstein=float(np.mean(wasserstein_vals)) if wasserstein_vals else float("nan"),
    )
