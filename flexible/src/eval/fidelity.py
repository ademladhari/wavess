"""Fidelity metrics (Sec. IV-A.3).

* FID: via clean-fid (preferred) or pytorch-fid fallback.
* NIQE / PIQE: via pyiqa.
* CLIP-score: alignment between text prompt and watermarked image using
  open_clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def _save_images_to_dir(images: Iterable[torch.Tensor], out_dir: Path) -> None:
    """Dump float (3, H, W) tensors in [0, 1] as .png files for clean-fid."""

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        arr = (img.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype("uint8")
        arr = np.transpose(arr, (1, 2, 0))
        Image.fromarray(arr).save(out_dir / f"{i:06d}.png")


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    device: torch.device | str = "cuda",
) -> float:
    """Compute FID between two directories of images."""

    try:
        from cleanfid import fid

        return float(fid.compute_fid(str(real_dir), str(fake_dir), device=str(device)))
    except Exception as e:
        print(f"[fid] clean-fid failed ({e}); falling back to pytorch-fid")
        from pytorch_fid.fid_score import calculate_fid_given_paths

        return float(
            calculate_fid_given_paths([str(real_dir), str(fake_dir)], 50, str(device), 2048)
        )


_IQA_NIQE = None
_IQA_PIQE = None


def _iqa_niqe(device: torch.device):
    global _IQA_NIQE
    if _IQA_NIQE is None:
        import pyiqa

        _IQA_NIQE = pyiqa.create_metric("niqe", device=device)
    return _IQA_NIQE


def _iqa_piqe(device: torch.device):
    global _IQA_PIQE
    if _IQA_PIQE is None:
        import pyiqa

        _IQA_PIQE = pyiqa.create_metric("piqe", device=device)
    return _IQA_PIQE


def compute_niqe(images: Sequence[torch.Tensor], device: torch.device | str = "cuda") -> float:
    device = torch.device(device)
    metric = _iqa_niqe(device)
    vals = []
    for x in images:
        x_batch = x.unsqueeze(0).to(device)
        vals.append(float(metric(x_batch).item()))
    return float(np.mean(vals))


def compute_piqe(images: Sequence[torch.Tensor], device: torch.device | str = "cuda") -> float:
    device = torch.device(device)
    metric = _iqa_piqe(device)
    vals = []
    for x in images:
        x_batch = x.unsqueeze(0).to(device)
        vals.append(float(metric(x_batch).item()))
    return float(np.mean(vals))


_CLIP_MODEL = None
_CLIP_TOKENIZER = None
_CLIP_PREPROCESS = None


def _load_clip(device: torch.device, model_name: str = "ViT-L-14", pretrained: str = "openai"):
    global _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval().to(device)
        _CLIP_MODEL = model
        _CLIP_TOKENIZER = open_clip.get_tokenizer(model_name)
        _CLIP_PREPROCESS = preprocess
    return _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_PREPROCESS


@torch.no_grad()
def compute_clip_score(
    images: Sequence[torch.Tensor],
    prompts: Sequence[str],
    device: torch.device | str = "cuda",
) -> float:
    device = torch.device(device)
    model, tokenizer, preprocess = _load_clip(device)

    from PIL import Image

    scores = []
    for img, prompt in zip(images, prompts):
        arr = (img.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype("uint8")
        pil = Image.fromarray(np.transpose(arr, (1, 2, 0)))
        x = preprocess(pil).unsqueeze(0).to(device)
        tok = tokenizer([prompt]).to(device)
        img_feat = model.encode_image(x)
        txt_feat = model.encode_text(tok)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        scores.append(float((img_feat * txt_feat).sum(dim=-1).item()))
    return float(np.mean(scores))
