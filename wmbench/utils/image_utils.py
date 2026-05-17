from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms


def normalize_tensor(images: torch.Tensor, norm_type: str) -> torch.Tensor:
    assert norm_type in ["imagenet", "naive"]
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean, std)
    else:
        raise AssertionError("unreachable")
    return torch.stack([normalize(image) for image in images])


def unnormalize_tensor(images: torch.Tensor, norm_type: str) -> torch.Tensor:
    assert norm_type in ["imagenet", "naive"]
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise AssertionError("unreachable")

    unnormalize = transforms.Normalize(
        (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
        (1 / std[0], 1 / std[1], 1 / std[2]),
    )
    return torch.stack([unnormalize(image) for image in images])


def to_tensor(images: list[Image.Image], norm_type: str | None = "naive") -> torch.Tensor:
    assert isinstance(images, list) and all(isinstance(image, Image.Image) for image in images)
    tensor = torch.stack([transforms.ToTensor()(image) for image in images])
    if norm_type is not None:
        tensor = normalize_tensor(tensor, norm_type)
    return tensor


def to_pil(images: torch.Tensor, norm_type: str | None = "naive") -> list[Image.Image]:
    assert isinstance(images, torch.Tensor)
    if norm_type is not None:
        images = unnormalize_tensor(images, norm_type).clamp(0, 1)
    return [transforms.ToPILImage()(image) for image in images.cpu()]


def renormalize_tensor(
    images: torch.Tensor,
    in_norm_type: str,
    out_norm_type: str,
) -> torch.Tensor:
    assert in_norm_type in ["imagenet", "naive"]
    assert out_norm_type in ["imagenet", "naive"]
    images = unnormalize_tensor(images, in_norm_type)
    images = normalize_tensor(images, out_norm_type)
    return images
