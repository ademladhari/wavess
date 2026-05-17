from __future__ import annotations

import torch
from PIL import Image

from wmbench.metrics.distributional import compute_fid


def compute_fid_metric(attacked: list[Image.Image], originals: list[Image.Image], **kwargs) -> float:
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    verbose = bool(kwargs.get("verbose", False))
    return compute_fid(originals, attacked, mode="clean", device=device, verbose=verbose)


def compute_clip_fid(attacked: list[Image.Image], originals: list[Image.Image], **kwargs) -> float:
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    verbose = bool(kwargs.get("verbose", False))
    return compute_fid(originals, attacked, mode="clip", device=device, verbose=verbose)


def compute_fid_from_dirs(reference_dir: str, candidate_dir: str, **kwargs) -> float:
    """Dataset-level FID using on-disk folders (stable cache keys in clean-fid)."""
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    verbose = bool(kwargs.get("verbose", False))
    return compute_fid(reference_dir, candidate_dir, mode="clean", device=device, verbose=verbose)


def compute_clip_fid_from_dirs(reference_dir: str, candidate_dir: str, **kwargs) -> float:
    device = kwargs.get("device") or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    verbose = bool(kwargs.get("verbose", False))
    return compute_fid(reference_dir, candidate_dir, mode="clip", device=device, verbose=verbose)
