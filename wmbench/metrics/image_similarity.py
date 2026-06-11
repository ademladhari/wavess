from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from wmbench.metrics.image import (
    compute_nmi_repeated,
    compute_psnr_repeated,
    compute_ssim_repeated,
)


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _pairs(originals: list[Image.Image], attacked: list[Image.Image]) -> tuple[list[Image.Image], list[Image.Image]]:
    if len(originals) != len(attacked):
        raise ValueError("originals and attacked must have same length")
    return originals, attacked


def compute_psnr(attacked: list[Image.Image], originals: list[Image.Image]) -> float:
    a, b = _pairs(originals, attacked)
    return _mean(compute_psnr_repeated(b, a, num_workers=os.cpu_count() or 4, verbose=False))


def compute_ssim(attacked: list[Image.Image], originals: list[Image.Image]) -> float:
    a, b = _pairs(originals, attacked)
    return _mean(compute_ssim_repeated(b, a, num_workers=os.cpu_count() or 4, verbose=False))


def compute_nmi(attacked: list[Image.Image], originals: list[Image.Image]) -> float:
    a, b = _pairs(originals, attacked)
    return _mean(compute_nmi_repeated(b, a, num_workers=os.cpu_count() or 4, verbose=False))


def _pil_batch_to_tensor(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    tensors = [TF.to_tensor(im.convert("RGB")) for im in images]
    return torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)


def _aligned_pair_batch_tensors(
    refs: list[Image.Image],
    cands: list[Image.Image],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack refs/cands for GPU batching; resize each candidate to its reference size."""
    ref_tensors = [TF.to_tensor(im.convert("RGB")) for im in refs]
    cand_tensors: list[torch.Tensor] = []
    for ref_t, cand_im in zip(ref_tensors, cands):
        cand_t = TF.to_tensor(cand_im.convert("RGB"))
        if cand_t.shape != ref_t.shape:
            cand_t = torch.nn.functional.interpolate(
                cand_t.unsqueeze(0),
                size=ref_t.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        cand_tensors.append(cand_t)
    ref = torch.stack(ref_tensors, dim=0).to(device=device, dtype=torch.float32)
    cand = torch.stack(cand_tensors, dim=0).to(device=device, dtype=torch.float32)
    return ref, cand


def _batched_psnr_per_image(ref: torch.Tensor, cand: torch.Tensor, *, max_val: float = 1.0) -> torch.Tensor:
    mse = ((ref - cand) ** 2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10((max_val**2) / mse.clamp(min=1e-10))


def _batched_ssim_per_image(ref: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    from kornia.metrics import ssim as kornia_ssim

    # kornia returns a map [B,C,H,W]; mean to one value per image (matches channel-mean SSIM spirit).
    return kornia_ssim(ref, cand, window_size=11, max_val=1.0).mean(dim=(1, 2, 3))


def compute_psnr_ssim_gpu(
    attacked: list[Image.Image],
    originals: list[Image.Image],
    *,
    device: torch.device,
    batch_size: int = 32,
    verbose: bool = False,
) -> tuple[float, float]:
    """Batched PSNR + SSIM on GPU (kornia SSIM). Falls back to CPU if device is CPU."""
    a, b = _pairs(originals, attacked)
    if device.type != "cuda":
        return compute_psnr(b, a), compute_ssim(b, a)

    batch_size = max(1, int(batch_size))
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    n = len(a)

    # Batch only images with the same reference resolution (mixed portrait/landscape datasets).
    from collections import defaultdict

    by_ref_size: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, ref_im in enumerate(a):
        by_ref_size[ref_im.size].append(i)

    batches: list[list[int]] = []
    for indices in by_ref_size.values():
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])

    it = batches
    if verbose:
        from tqdm.auto import tqdm

        it = tqdm(it, total=len(batches), desc="PSNR+SSIM")

    for idxs in it:
        ref_imgs = [a[i] for i in idxs]
        cand_imgs = [b[i] for i in idxs]
        ref, cand = _aligned_pair_batch_tensors(ref_imgs, cand_imgs, device)
        psnr_vals.extend(_batched_psnr_per_image(ref, cand).detach().cpu().tolist())
        ssim_vals.extend(_batched_ssim_per_image(ref, cand).detach().cpu().tolist())

    return _mean(psnr_vals), _mean(ssim_vals)
