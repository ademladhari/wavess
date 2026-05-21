"""ResNet18-based watermark extractor baseline (Sec. IV-F / Table V).

The paper contrasts their Transformer+MLP extractor against a ResNet18
extractor trained under identical settings. This module provides that
baseline: ResNet18 backbone + linear head reshaped to (4, 64, 64).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Extractor(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        backbone = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        backbone.fc = nn.Identity()
        self.backbone = backbone

        latent_dim = latent_channels * latent_size * latent_size
        self.head = nn.Linear(512, latent_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        feat = self.backbone(image)
        z = self.head(feat)
        return z.view(b, self.latent_channels, self.latent_size, self.latent_size)
