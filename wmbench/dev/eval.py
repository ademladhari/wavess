from __future__ import annotations

import numpy as np
from sklearn import metrics


def bit_error_rate(pred: np.ndarray, target: np.ndarray) -> float:
    if not pred.dtype == target.dtype == bool:
        raise ValueError(f"Cannot compute BER for {pred.dtype} and {target.dtype}")
    return float(np.mean(pred != target))


def complex_l1(pred: np.ndarray, target: np.ndarray) -> float:
    if not pred.dtype == target.dtype == np.float16:
        raise ValueError(f"Cannot compute Complex L1 for {pred.dtype} and {target.dtype}")
    pred = pred.astype(np.float32).reshape(2, -1)
    target = target.astype(np.float32).reshape(2, -1)
    return float(np.sqrt(((pred - target) ** 2).sum(0)).mean())


def message_distance(pred: np.ndarray, target: np.ndarray) -> float:
    if target.dtype == bool:
        return bit_error_rate(pred, target)
    if target.dtype == np.float16:
        return complex_l1(pred, target)
    raise TypeError


def detection_perforamance(original_distances: list[float], watermarked_distances: list[float]):
    if len(original_distances) != len(watermarked_distances):
        raise ValueError("Length of distances must be equal")
    y_true = [0] * len(original_distances) + [1] * len(watermarked_distances)
    y_score = (-np.array(original_distances + watermarked_distances)).tolist()
    fpr, tpr, _thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    acc_1 = float(np.max(1 - (fpr + (1 - tpr)) / 2))
    auc_1 = float(metrics.auc(fpr, tpr))
    low100_1 = float(tpr[np.where(fpr < 0.01)[0][-1]])
    low1000_1 = float(tpr[np.where(fpr < 0.001)[0][-1]])
    return {
        "acc_1": acc_1,
        "auc_1": auc_1,
        "low100_1": low100_1,
        "low1000_1": low1000_1,
    }


def detection_performance(original_distances: list[float], watermarked_distances: list[float]):
    return detection_perforamance(original_distances, watermarked_distances)


def mean_and_std(values):
    if values is None:
        return None
    return float(np.mean(values)), float(np.std(values))


def combine_means_and_stds(mean_and_std1, mean_and_std2):
    if mean_and_std1 is None or mean_and_std2 is None:
        return None
    mean1, std1 = mean_and_std1
    mean2, std2 = mean_and_std2
    mean = (mean1 + mean2) / 2
    std = float(np.sqrt((std1**2 + std2**2) / 2))
    return float(mean), std
