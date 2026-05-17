from __future__ import annotations

import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .lpips import LPIPS

try:
    from .watson import LossProvider
except (ModuleNotFoundError, ImportError):
    try:
        from .metrics.watson import LossProvider
    except (ModuleNotFoundError, ImportError):
        LossProvider = None


def _to_tensor(images: list[Image.Image]) -> torch.Tensor:
    """Convert list of PIL Images to a batched tensor [B, C, H, W] in [0, 1]."""
    import numpy as np
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    tensors = [transform(img) for img in images]
    return torch.stack(tensors, dim=0)


def _pad_images_to_multiple(images: list[Image.Image], multiple: int = 8) -> list[Image.Image]:
    """Pad images so H/W are divisible by `multiple` (Watson FFT requirement)."""
    padded: list[Image.Image] = []
    for img in images:
        w, h = img.size
        pad_w = (multiple - (w % multiple)) % multiple
        pad_h = (multiple - (h % multiple)) % multiple
        if pad_w == 0 and pad_h == 0:
            padded.append(img)
            continue
        # Pad on right/bottom to keep top-left alignment stable
        padded.append(ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0))
    return padded


def _align_pair(reference: Image.Image, candidate: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Align candidate shape/mode to reference for pairwise perceptual metrics."""
    ref = reference
    cand = candidate
    if cand.mode != ref.mode:
        cand = cand.convert(ref.mode)
    if cand.size != ref.size:
        cand = cand.resize(ref.size, Image.Resampling.BICUBIC)
    return ref, cand


def load_perceptual_models(
    metric_name: str,
    mode: str,
    device: torch.device | None = None,
) -> LPIPS | LossProvider:
    """Load perceptual model (LPIPS or Watson).

    Parameters:
    - metric_name: "lpips" or "watson"
    - mode: for LPIPS: "vgg" or "alex"; for Watson: "vgg", "dft", or "fft"
    - device: torch device (defaults to cuda if available)
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    assert metric_name in ["lpips", "watson"], f"Unknown metric: {metric_name}"

    if metric_name == "lpips":
        assert mode in ["vgg", "alex"], f"LPIPS mode must be 'vgg' or 'alex', got {mode}"
        perceptual_model = LPIPS(net=mode).to(device)
    elif metric_name == "watson":
        assert mode in ["vgg", "dft", "fft"], f"Watson mode must be 'vgg', 'dft', or 'fft', got {mode}"
        if LossProvider is None:
            raise RuntimeError(
                "Watson perceptual model not available; ensure watson dependencies are installed."
            )
        perceptual_model = LossProvider().get_loss_function(
            "Watson-" + mode,
            colorspace="RGB",
            pretrained=True,
            reduction="none",
        ).to(device)
    else:
        raise AssertionError(f"Unknown metric: {metric_name}")

    return perceptual_model


def compute_metric(
    image1: Image.Image,
    image2: Image.Image,
    perceptual_model,
    device: torch.device | None = None,
) -> float:
    """Compute perceptual metric between two images."""
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)
    image1_tensor = _to_tensor([image1]).to(device)
    image2_tensor = _to_tensor([image2]).to(device)
    return float(perceptual_model(image1_tensor, image2_tensor).cpu().item())


def compute_lpips(
    image1: Image.Image,
    image2: Image.Image,
    mode: str = "alex",
    device: torch.device | None = None,
) -> float:
    """Compute LPIPS distance between two images."""
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    perceptual_model = load_perceptual_models("lpips", mode, device)
    return compute_metric(image1, image2, perceptual_model, device)


def compute_watson(
    image1: Image.Image,
    image2: Image.Image,
    mode: str = "dft",
    device: torch.device | None = None,
) -> float:
    """Compute Watson distance between two images."""
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    perceptual_model = load_perceptual_models("watson", mode, device)
    return compute_metric(image1, image2, perceptual_model, device)


def compute_perceptual_metric_repeated(
    images1: list[Image.Image],
    images2: list[Image.Image],
    metric_name: str,
    mode: str,
    model=None,
    device: torch.device | None = None,
    verbose: bool = False,
    batch_size: int = 1,
) -> list[float]:
    """Compute perceptual metrics between pairs of images.

    Parameters:
    - images1: list of PIL Images (reference)
    - images2: list of PIL Images (test)
    - metric_name: "lpips" or "watson"
    - mode: model variant (e.g., "alex", "dft")
    - model: pre-loaded model (optional; will be loaded if None)
    - device: torch device (defaults to cuda if available)

    Returns: list of metric values
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
    assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
    assert len(images1) == len(images2)

    if model is None:
        model = load_perceptual_models(metric_name, mode, device)

    batch_size = max(1, int(batch_size))
    values: list[float] = [0.0] * len(images1)

    def _run_batch(ref_batch: list[Image.Image], cand_batch: list[Image.Image]) -> list[float]:
        if metric_name == "watson":
            ref_batch = _pad_images_to_multiple(ref_batch, multiple=8)
            cand_batch = _pad_images_to_multiple(cand_batch, multiple=8)
        tensor1 = _to_tensor(ref_batch).to(device)
        tensor2 = _to_tensor(cand_batch).to(device)
        out = model(tensor1, tensor2).detach().cpu().numpy().flatten().tolist()
        return [float(v) for v in out]

    # Bucket by aligned size/mode so stacked tensor batches are valid.
    buckets: dict[tuple[str, tuple[int, int]], list[tuple[int, Image.Image, Image.Image]]] = {}
    pbar = tqdm(total=len(images1), desc=f"{metric_name.upper()} ") if verbose else None
    for i, (ref, cand) in enumerate(zip(images1, images2)):
        ref_aligned, cand_aligned = _align_pair(ref, cand)
        sig = (ref_aligned.mode, ref_aligned.size)
        bucket = buckets.setdefault(sig, [])
        bucket.append((i, ref_aligned, cand_aligned))
        if len(bucket) >= batch_size:
            idxs = [it[0] for it in bucket]
            refs = [it[1] for it in bucket]
            cands = [it[2] for it in bucket]
            out_vals = _run_batch(refs, cands)
            for j, v in zip(idxs, out_vals):
                values[j] = v
            if pbar is not None:
                pbar.update(len(bucket))
            buckets[sig] = []

    for bucket in buckets.values():
        if not bucket:
            continue
        idxs = [it[0] for it in bucket]
        refs = [it[1] for it in bucket]
        cands = [it[2] for it in bucket]
        out_vals = _run_batch(refs, cands)
        for j, v in zip(idxs, out_vals):
            values[j] = v
        if pbar is not None:
            pbar.update(len(bucket))
    if pbar is not None:
        pbar.close()
    return values


def compute_lpips_repeated(
    images1: list[Image.Image],
    images2: list[Image.Image],
    mode: str = "alex",
    model=None,
    device: torch.device | None = None,
    verbose: bool = False,
    batch_size: int = 1,
) -> list[float]:
    """Compute LPIPS distances between pairs of images."""
    return compute_perceptual_metric_repeated(
        images1, images2, "lpips", mode, model, device, verbose, batch_size
    )


def compute_watson_repeated(
    images1: list[Image.Image],
    images2: list[Image.Image],
    mode: str = "dft",
    model=None,
    device: torch.device | None = None,
    verbose: bool = False,
    batch_size: int = 1,
) -> list[float]:
    """Compute Watson distances between pairs of images."""
    return compute_perceptual_metric_repeated(
        images1, images2, "watson", mode, model, device, verbose, batch_size
    )
