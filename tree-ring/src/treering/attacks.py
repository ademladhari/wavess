from __future__ import annotations

from typing import Any

import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _crop_rescale(image: Image.Image, crop_ratio: float) -> Image.Image:
    width, height = image.size
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def apply_attack(image: Image.Image, attack: str, **kwargs: Any) -> Image.Image:
    attack_name = attack.lower()
    if attack_name == "none":
        return image
    if attack_name == "rotation":
        angle = float(kwargs.get("angle", 75.0))
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    if attack_name == "jpeg":
        quality = int(kwargs.get("quality", 25))
        return _jpeg(image, quality=quality)
    if attack_name == "crop_rescale":
        crop_ratio = float(kwargs.get("crop_ratio", 0.75))
        return _crop_rescale(image, crop_ratio=crop_ratio)
    if attack_name == "gaussian_blur":
        radius = float(kwargs.get("radius", 2.0))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if attack_name == "gaussian_noise":
        sigma = float(kwargs.get("sigma", 0.1))
        tensor = TF.to_tensor(image).clamp(0, 1)
        noisy = (tensor + sigma * torch.randn_like(tensor)).clamp(0, 1)
        return TF.to_pil_image(noisy)
    if attack_name == "color_jitter":
        brightness = float(kwargs.get("brightness", 6.0))
        return TF.adjust_brightness(image, brightness_factor=brightness)
    raise ValueError(f"Unsupported attack '{attack}'.")
