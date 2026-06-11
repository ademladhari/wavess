"""PSNR / SSIM / bit accuracy / detection metrics."""

from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn import metrics


def psnr_ssim_pair(ref: Image.Image, cand: Image.Image) -> tuple[float, float]:
    from wmbench.metrics.image import compute_psnr, compute_ssim

    return compute_psnr(ref, cand), compute_ssim(ref, cand)


def detection_auroc_and_tpr(
    pos_scores: np.ndarray, neg_scores: np.ndarray, fpr_target: float = 0.01
) -> tuple[float, float]:
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan"), float("nan")
    y_true = np.concatenate(
        [np.zeros(neg_scores.size, dtype=np.int32), np.ones(pos_scores.size, dtype=np.int32)]
    )
    y_score = np.concatenate([neg_scores, pos_scores])
    auroc = float(metrics.roc_auc_score(y_true, y_score))
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    below = np.where(fpr < fpr_target)[0]
    tpr_at = float(tpr[below[-1]]) if below.size else float(tpr[0])
    return auroc, tpr_at
