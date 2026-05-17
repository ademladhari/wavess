from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

METRIC_KEYS: tuple[str, ...] = (
    "PSNR",
    "SSIM",
    "NMI",
    "LPIPS",
    "FID",
    "CLIP_FID",
    "aesthetics_delta",
    "artifacts",
)

# invert=True for PSNR, SSIM, NMI (higher raw = better fidelity)
INVERT_METRICS: dict[str, bool] = {
    "PSNR": True,
    "SSIM": True,
    "NMI": True,
    "LPIPS": False,
    "FID": False,
    "CLIP_FID": False,
    "aesthetics_delta": False,
    "artifacts": False,
}


def compute_anchors(all_raw_metrics: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    """Return p10, p90 per metric name over the full pool."""
    anchors: dict[str, tuple[float, float]] = {}
    for name, values in all_raw_metrics.items():
        clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if len(clean) < 2:
            anchors[name] = (0.0, 1.0)
            continue
        xs = sorted(clean)
        p10 = xs[int(0.10 * (len(xs) - 1))]
        p90 = xs[int(0.90 * (len(xs) - 1))]
        anchors[name] = (float(p10), float(p90))
    return anchors


def normalize_metric(value: float, p10: float, p90: float, invert: bool) -> float:
    """Linear map: p10 → 0.1, p90 → 0.9 along raw scale; then optional orientation.

    invert=True: keep orientation (higher raw → higher normalized) for PSNR/SSIM/NMI.
    invert=False: flip so higher raw → lower normalized (LPIPS, FID, …).
    """
    if math.isnan(value) or p90 - p10 < 1e-12:
        return 0.0
    t = (value - p10) / (p90 - p10)
    scaled = 0.1 + t * 0.8
    if not invert:
        scaled = 1.0 - scaled
    return float(max(0.0, min(1.0, scaled)))


def compute_Q(normalized_metrics: dict[str, float]) -> float:
    vals = [normalized_metrics[k] for k in METRIC_KEYS if k in normalized_metrics]
    return float(sum(vals) / max(len(vals), 1))


def load_or_compute_anchors(
    anchor_path: str,
    all_raw_metrics: dict[str, list[float]],
) -> dict[str, tuple[float, float]]:
    if os.path.isfile(anchor_path):
        with open(anchor_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return {k: (float(v[0]), float(v[1])) for k, v in data.items()}
    anchors = compute_anchors(all_raw_metrics)
    d = os.path.dirname(anchor_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(anchor_path, "w", encoding="utf-8") as f:
        json.dump({k: [a, b] for k, (a, b) in anchors.items()}, f, indent=2)
    return anchors


def normalize_row(raw_row: dict[str, float], anchors: dict[str, tuple[float, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in METRIC_KEYS:
        if k not in raw_row:
            continue
        p10, p90 = anchors.get(k, (0.0, 1.0))
        out[k] = normalize_metric(float(raw_row[k]), p10, p90, INVERT_METRICS.get(k, False))
    return out


def q_from_raw_row(raw_row: dict[str, float], anchors: dict[str, tuple[float, float]]) -> float:
    return compute_Q(normalize_row(raw_row, anchors))


def interp_q_at_p(
    strengths: list[float],
    p_values: list[float],
    q_values: list[float],
    target_p: float,
) -> float:
    """Q interpolated at P=target_p; -inf if P at weakest strength < target_p; +inf if all P > target_p."""
    pts = [
        (float(s), float(p), float(q))
        for s, p, q in zip(strengths, p_values, q_values)
        if not math.isnan(p) and not math.isnan(q)
    ]
    if not pts:
        return float("nan")
    idx_w = min(range(len(pts)), key=lambda i: pts[i][0])
    if pts[idx_w][1] < target_p - 1e-9:
        return float("-inf")
    p_only = [t[1] for t in pts]
    if min(p_only) > target_p + 1e-9:
        return float("inf")
    pq = sorted([(t[1], t[2]) for t in pts], key=lambda x: x[0])
    p_s = [a for a, _ in pq]
    q_s = [b for _, b in pq]
    if len(p_s) == 1:
        return q_s[0]
    return float(np.interp(target_p, p_s, q_s))
