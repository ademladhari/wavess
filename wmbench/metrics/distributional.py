from __future__ import annotations

import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .clean_fid import fid

# (ref_dir_abs_norm, mode, model_name, device) -> (mu, sigma); avoids fid_folder -> get_reference_statistics HTTP checks
_ref_dir_moments: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
# (mode, model_name, device) -> (feature_model, custom_resize_fn)
_feat_model_cache: dict[tuple[str, str, str], tuple[object, object | None]] = {}


def _matrix_sqrt_psd_torch(S: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition (GPU-friendly)."""
    w, Q = torch.linalg.eigh(S)
    w = torch.clamp(w, min=0)
    rt = torch.sqrt(w)
    return (Q * rt.unsqueeze(0)) @ Q.T


def _frechet_distance_gaussians_cuda(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    *,
    device: torch.device,
    eps: float = 1e-6,
) -> float:
    """Same Fréchet term as SciPy FID, but avoids scipy.linalg.sqrtm on huge dense covariances.

    Uses the identity tr(sqrt(S1 @ S2)) == tr(sqrt(S1^{1/2} @ S2 @ S1^{1/2})) for PSD covariances,
    with only PSD matrix square roots (eigh). Much faster than sqrtm(S1@S2) on CPU for dim ~2048.
    """
    mu1_t = torch.as_tensor(mu1, device=device, dtype=torch.float64).reshape(-1)
    mu2_t = torch.as_tensor(mu2, device=device, dtype=torch.float64).reshape(-1)
    n = int(mu1_t.shape[0])
    eye = torch.eye(n, device=device, dtype=torch.float64)
    s1 = torch.as_tensor(sigma1, device=device, dtype=torch.float64)
    s2 = torch.as_tensor(sigma2, device=device, dtype=torch.float64)
    s1_r = s1 + eps * eye
    s2_r = s2 + eps * eye
    diff = mu1_t - mu2_t
    sqrt_s1 = _matrix_sqrt_psd_torch(s1_r)
    t_mat = sqrt_s1 @ s2_r @ sqrt_s1
    sqrt_t = _matrix_sqrt_psd_torch(t_mat)
    fd = diff.dot(diff) + torch.trace(s1) + torch.trace(s2) - 2 * torch.trace(sqrt_t)
    return float(fd.detach().cpu())


def save_single_image_to_temp(i: int, image: Image.Image, temp_dir: str) -> None:
    save_path = os.path.join(temp_dir, f"{i}.png")
    image.save(save_path, "PNG")


def save_images_to_temp(images: list[Image.Image], num_workers: int, verbose: bool = False) -> str:
    assert isinstance(images, list) and isinstance(images[0], Image.Image)
    temp_dir = tempfile.mkdtemp()

    # ThreadPool avoids Windows spawn + PIL pickling failures seen with ProcessPoolExecutor.
    workers = max(1, min(num_workers, len(images), (os.cpu_count() or 4)))
    func = partial(save_single_image_to_temp, temp_dir=temp_dir)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = executor.map(func, range(len(images)), images)
        list(tasks) if not verbose else list(
            tqdm(
                tasks,
                total=len(images),
                desc="Saving images ",
            )
        )
    return temp_dir


def _feat_model_for_fid(
    mode: str,
    model_name: str,
    device: torch.device,
    *,
    use_dataparallel: bool = True,
):
    if model_name == "inception_v3":
        feat_model = fid.build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
        return feat_model, None
    if model_name == "clip_vit_b_32":
        from .clean_fid.clip_features import CLIP_fx, img_preprocess_clip

        feat_model = CLIP_fx("ViT-B/32", device=device)
        return feat_model, img_preprocess_clip
    raise ValueError(f"Unknown FID model_name: {model_name}")


def _get_cached_feat_model_for_fid(
    mode: str,
    model_name: str,
    device: torch.device,
):
    """Reuse expensive feature extractors (especially CLIP) across metric cells."""
    key = (mode, model_name, str(device))
    cached = _feat_model_cache.get(key)
    if cached is not None:
        return cached
    built = _feat_model_for_fid(mode, model_name, device)
    _feat_model_cache[key] = built
    return built


def _fid_between_dirs(
    path1: str,
    path2: str,
    *,
    mode: str,
    model_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    verbose: bool,
) -> float:
    feat_model, custom_fn_resize = _get_cached_feat_model_for_fid(mode, model_name, device)
    cache_key = (
        os.path.normcase(os.path.abspath(path1)),
        mode,
        model_name,
        str(device),
    )
    if cache_key not in _ref_dir_moments:
        ref_feats = fid.get_folder_features(
            path1,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            custom_fn_resize=custom_fn_resize,
            description=f"FID ref {os.path.basename(path1.rstrip(os.sep))} : ",
            verbose=verbose,
        )
        _ref_dir_moments[cache_key] = (
            np.mean(ref_feats, axis=0),
            np.cov(ref_feats, rowvar=False),
        )
    mu1, sigma1 = _ref_dir_moments[cache_key]
    cand_feats = fid.get_folder_features(
        path2,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        custom_fn_resize=custom_fn_resize,
        description=f"FID {os.path.basename(path2.rstrip(os.sep))} : ",
        verbose=verbose,
    )
    mu2 = np.mean(cand_feats, axis=0)
    sigma2 = np.cov(cand_feats, rowvar=False)
    # tqdm above ends when features are extracted; the remaining cost is mostly Fréchet linear
    # algebra. scipy.linalg.sqrtm on (2048×2048) is very slow on CPU — use CUDA eigh when possible.
    if device.type == "cuda":
        try:
            out = _frechet_distance_gaussians_cuda(mu1, sigma1, mu2, sigma2, device=device)
            if np.isfinite(out):
                return float(out)
        except Exception:
            pass
    return float(fid.frechet_distance(mu1, sigma1, mu2, sigma2))


def compute_fid(
    images1: str | list[Image.Image],
    images2: str | list[Image.Image],
    mode: str = "legacy",
    device: torch.device | None = None,
    batch_size: int = 64,
    num_workers: int | None = None,
    verbose: bool = False,
) -> float:
    """Compute FID score between two sets of images.

    Parameters:
    - images1: directory path or list of PIL Images (reference/clean)
    - images2: directory path or list of PIL Images (test/attacked)
    - mode: "legacy", "clean", or "clip"
    - device: torch device (defaults to cuda if available)
    - batch_size: batch size for FID computation
    - num_workers: number of workers (defaults based on GPU count)
    - verbose: whether to show progress bars
    """
    assert mode in ["legacy", "clean", "clip"]
    if mode == "legacy":
        mode = "legacy_pytorch"
        model_name = "inception_v3"
    elif mode == "clean":
        mode = "clean"
        model_name = "inception_v3"
    elif mode == "clip":
        mode = "clean"
        model_name = "clip_vit_b_32"
    else:
        assert False

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if num_workers is not None:
        assert 0 <= num_workers <= (os.cpu_count() or 1)
    else:
        # Windows + DataLoader workers often breaks nested multiprocessing; default to 0 there.
        if sys.platform == "win32":
            num_workers = 0
        else:
            num_workers = max(torch.cuda.device_count() * 4, 8)

    temp_dirs: list[str] = []
    try:
        # Handle paths or PIL image lists
        if not isinstance(images1, (str, Path)):
            assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
            assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
            path1 = save_images_to_temp(images1, num_workers=num_workers, verbose=verbose)
            path2 = save_images_to_temp(images2, num_workers=num_workers, verbose=verbose)
            temp_dirs.extend([path1, path2])
        else:
            assert isinstance(images1, (str, Path)) and os.path.exists(images1)
            assert isinstance(images2, (str, Path)) and os.path.exists(images2)
            path1 = str(images1)
            path2 = str(images2)

        # Folder pairs: compute Frechet distance from features only. Do not use fid_folder, which pulls
        # get_reference_statistics() and may hit HTTP "downloading statistics" every call for custom names.
        return _fid_between_dirs(
            path1,
            path2,
            mode=mode,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )
    finally:
        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except OSError:
                pass
