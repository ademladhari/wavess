from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from wmbench.utils.exp_utils import set_random_seed
from wmbench.utils.image_utils import to_pil, to_tensor


distortion_strength_paras = dict(
    rotation=(0, 45),
    resizedcrop=(1, 0.5),
    erasing=(0, 0.25),
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(90, 10),
)


def relative_strength_to_absolute(strength: float, distortion_type: str) -> float:
    assert 0 <= strength <= 1
    strength = (
        strength
        * (distortion_strength_paras[distortion_type][1] - distortion_strength_paras[distortion_type][0])
        + distortion_strength_paras[distortion_type][0]
    )
    strength = max(strength, min(*distortion_strength_paras[distortion_type]))
    strength = min(strength, max(*distortion_strength_paras[distortion_type]))
    return float(strength)


def apply_distortion(
    images: list[Image.Image] | torch.Tensor,
    distortion_type: str,
    strength: float | None = None,
    distortion_seed: int = 0,
    same_operation: bool = False,
    relative_strength: bool = True,
    return_image: bool = True,
):
    if not isinstance(images[0], Image.Image):
        images = to_pil(images)  # type: ignore[arg-type]

    if relative_strength:
        if strength is None:
            raise ValueError("strength must be provided when relative_strength=True")
        strength = relative_strength_to_absolute(strength, distortion_type)

    distorted_images: list[Image.Image] = []
    seed = distortion_seed
    for image in images:  # type: ignore[assignment]
        distorted_images.append(apply_single_distortion(image, distortion_type, strength, distortion_seed=seed))
        if not same_operation:
            seed += 1

    if not return_image:
        return to_tensor(distorted_images)
    return distorted_images


def apply_single_distortion(
    image: Image.Image,
    distortion_type: str,
    strength: float | None = None,
    distortion_seed: int = 0,
) -> Image.Image:
    assert isinstance(image, Image.Image)
    set_random_seed(distortion_seed)
    assert distortion_type in distortion_strength_paras

    if strength is not None:
        assert min(*distortion_strength_paras[distortion_type]) <= strength <= max(*distortion_strength_paras[distortion_type])

    if distortion_type == "rotation":
        angle = strength if strength is not None else random.uniform(*distortion_strength_paras["rotation"])
        distorted_image = F.rotate(image, angle)

    elif distortion_type == "resizedcrop":
        scale = strength if strength is not None else random.uniform(*distortion_strength_paras["resizedcrop"])
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(scale, scale), ratio=(1, 1))
        distorted_image = F.resized_crop(image, i, j, h, w, image.size)

    elif distortion_type == "erasing":
        scale = strength if strength is not None else random.uniform(*distortion_strength_paras["erasing"])
        image_t = to_tensor([image], norm_type=None)
        i, j, h, w, v = T.RandomErasing.get_params(image_t, scale=(scale, scale), ratio=(1, 1), value=[0])
        distorted_tensor = F.erase(image_t, i, j, h, w, v)
        distorted_image = to_pil(distorted_tensor, norm_type=None)[0]

    elif distortion_type == "brightness":
        factor = strength if strength is not None else random.uniform(*distortion_strength_paras["brightness"])
        distorted_image = ImageEnhance.Brightness(image).enhance(factor)

    elif distortion_type == "contrast":
        factor = strength if strength is not None else random.uniform(*distortion_strength_paras["contrast"])
        distorted_image = ImageEnhance.Contrast(image).enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = int(strength) if strength is not None else random.uniform(*distortion_strength_paras["blurring"])
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = strength if strength is not None else random.uniform(*distortion_strength_paras["noise"])
        image_t = to_tensor([image], norm_type=None)
        noise = torch.randn(image_t.size()) * std
        distorted_image = to_pil((image_t + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = strength if strength is not None else random.uniform(*distortion_strength_paras["compression"])
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)

    else:
        raise AssertionError("unreachable")

    return distorted_image
