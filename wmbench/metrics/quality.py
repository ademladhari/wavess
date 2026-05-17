from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from wmbench.metrics.aesthetics import compute_aesthetics_and_artifacts_scores, load_aesthetics_and_artifacts_models
from wmbench.metrics.perceptual import compute_lpips_repeated


def compute_lpips(attacked: list[Image.Image], originals: list[Image.Image], **kwargs) -> float:
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = kwargs.get("model")
    verbose = bool(kwargs.get("verbose", False))
    batch_size = int(kwargs.get("batch_size", 1) or 1)
    vals = compute_lpips_repeated(
        originals,
        attacked,
        mode="alex",
        model=model,
        device=device,
        verbose=verbose,
        batch_size=batch_size,
    )
    return float(np.mean(vals)) if vals else float("nan")


def compute_aesthetics_delta(attacked: list[Image.Image], originals: list[Image.Image], **kwargs) -> float:
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    models = kwargs.get("aesthetics_models")
    if models is None:
        models = load_aesthetics_and_artifacts_models(device=device)
    r0, _a0 = compute_aesthetics_and_artifacts_scores(originals, models, device=device)
    r1, _a1 = compute_aesthetics_and_artifacts_scores(attacked, models, device=device)
    deltas = [float(o) - float(a) for o, a in zip(r0, r1)]
    return float(np.mean(deltas)) if deltas else float("nan")


def compute_artifacts(attacked: list[Image.Image], originals: list[Image.Image], **kwargs) -> float:
    del originals
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    models = kwargs.get("aesthetics_models")
    if models is None:
        models = load_aesthetics_and_artifacts_models(device=device)
    _r, art = compute_aesthetics_and_artifacts_scores(attacked, models, device=device)
    return float(np.mean(art)) if art else float("nan")


def compute_aesthetics_delta_and_artifacts(
    attacked: list[Image.Image], originals: list[Image.Image], **kwargs
) -> tuple[float, float]:
    """Compute aesthetics delta + artifacts together to avoid duplicate model passes."""
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    models = kwargs.get("aesthetics_models")
    original_ratings = kwargs.get("original_ratings")
    if models is None:
        models = load_aesthetics_and_artifacts_models(device=device)
    if original_ratings is None:
        r0, _a0 = compute_aesthetics_and_artifacts_scores(originals, models, device=device)
    else:
        r0 = [float(v) for v in original_ratings]
    r1, art = compute_aesthetics_and_artifacts_scores(attacked, models, device=device)
    deltas = [float(o) - float(a) for o, a in zip(r0, r1)]
    delta_val = float(np.mean(deltas)) if deltas else float("nan")
    art_val = float(np.mean(art)) if art else float("nan")
    return delta_val, art_val
