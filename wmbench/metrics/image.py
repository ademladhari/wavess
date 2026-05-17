from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    normalized_mutual_information,
    peak_signal_noise_ratio,
    structural_similarity as structural_similarity_index_measure,
)
from tqdm.auto import tqdm


def convert_image_pair_to_numpy(image1: Image.Image, image2: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(image1, Image.Image) and isinstance(image2, Image.Image)
    if image2.mode != image1.mode:
        image2 = image2.convert(image1.mode)
    if image2.size != image1.size:
        image2 = image2.resize(image1.size, Image.Resampling.BICUBIC)

    image1_np = np.array(image1)
    image2_np = np.array(image2)
    assert image1_np.shape == image2_np.shape

    return image1_np, image2_np


def compute_mse(image1: Image.Image, image2: Image.Image) -> float:
    image1_np, image2_np = convert_image_pair_to_numpy(image1, image2)
    return float(mean_squared_error(image1_np, image2_np))


def compute_psnr(image1: Image.Image, image2: Image.Image) -> float:
    image1_np, image2_np = convert_image_pair_to_numpy(image1, image2)
    return float(peak_signal_noise_ratio(image1_np, image2_np))


def compute_ssim(image1: Image.Image, image2: Image.Image) -> float:
    image1_np, image2_np = convert_image_pair_to_numpy(image1, image2)
    if image1_np.ndim == 2:
        return float(structural_similarity_index_measure(image1_np, image2_np))
    return float(structural_similarity_index_measure(image1_np, image2_np, channel_axis=2))


def compute_nmi(image1: Image.Image, image2: Image.Image) -> float:
    image1_np, image2_np = convert_image_pair_to_numpy(image1, image2)
    return float(normalized_mutual_information(image1_np, image2_np))


def compute_metric_repeated(
    images1: list[Image.Image],
    images2: list[Image.Image],
    metric_func,
    num_workers: int | None = None,
    verbose: bool = False,
):
    assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
    assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
    assert len(images1) == len(images2)

    if num_workers is not None:
        assert 1 <= num_workers <= os.cpu_count()
    else:
        num_workers = max(torch.cuda.device_count() * 4, 8)

    metric_name = metric_func.__name__.split("_")[1].upper()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = executor.map(metric_func, images1, images2)
        values = list(tasks) if not verbose else list(
            tqdm(tasks, total=len(images1), desc=f"{metric_name} ")
        )

    return values


def compute_mse_repeated(images1, images2, num_workers=None, verbose: bool = False):
    return compute_metric_repeated(images1, images2, compute_mse, num_workers, verbose)


def compute_psnr_repeated(images1, images2, num_workers=None, verbose: bool = False):
    return compute_metric_repeated(images1, images2, compute_psnr, num_workers, verbose)


def compute_ssim_repeated(images1, images2, num_workers=None, verbose: bool = False):
    return compute_metric_repeated(images1, images2, compute_ssim, num_workers, verbose)


def compute_nmi_repeated(images1, images2, num_workers=None, verbose: bool = False):
    return compute_metric_repeated(images1, images2, compute_nmi, num_workers, verbose)


def compute_image_distance_repeated(
    images1: list[Image.Image],
    images2: list[Image.Image],
    metric_name: str,
    num_workers: int | None = None,
    verbose: bool = False,
):
    metric_func = {"psnr": compute_psnr, "ssim": compute_ssim, "nmi": compute_nmi}[metric_name]
    return compute_metric_repeated(images1, images2, metric_func, num_workers, verbose)
