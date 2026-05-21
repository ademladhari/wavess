from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import auc, roc_curve


@dataclass(slots=True)
class RocMetrics:
    auc: float
    tpr_at_target_fpr: float
    target_fpr: float


def compute_roc_metrics(
    positive_scores: list[float],
    negative_scores: list[float],
    target_fpr: float = 0.01,
) -> RocMetrics:
    y_true = np.array([1] * len(positive_scores) + [0] * len(negative_scores), dtype=np.int32)
    scores = np.array(positive_scores + negative_scores, dtype=np.float64)

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    tpr_at_target = float(np.interp(target_fpr, fpr, tpr))
    return RocMetrics(auc=roc_auc, tpr_at_target_fpr=tpr_at_target, target_fpr=target_fpr)
