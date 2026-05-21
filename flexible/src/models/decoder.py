"""Watermark Decoder D (Fig. 3 / Sec. III-C).

Paper description (Sec. III-C, p. 13954-13955):
    "D is made up of a linear layer, four pairs of combined layers,
     consisting of a transposed convolutional layer, a batch normalization
     layer, and a ReLU layer, a convolutional layer, and a fully connected
     layer, which is the function of sigmoid(.). Finally, the watermark wo
     is reconstructed."

Loss (eq. 3): L_r = MSE(wo, w).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkDecoder(nn.Module):
    """Latent zT (4 x 64 x 64) -> reconstructed n-bit watermark wo in [0,1]."""

    def __init__(
        self,
        n_bits: int,
        latent_channels: int = 4,
        latent_size: int = 64,
        base_channels: int = 64,
        tconv_blocks: int = 4,
        feature_hw: int = 8,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.base_channels = base_channels
        self.feature_hw = feature_hw

        latent_dim = latent_channels * latent_size * latent_size
        feat_dim = base_channels * feature_hw * feature_hw
        self.project = nn.Linear(latent_dim, feat_dim)

        blocks = []
        for _ in range(tconv_blocks):
            blocks.append(
                nn.ConvTranspose2d(
                    base_channels,
                    base_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                )
            )
            blocks.append(nn.BatchNorm2d(base_channels))
            blocks.append(nn.ReLU(inplace=True))
        self.tconv_blocks = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.head = nn.Linear(feat_dim, n_bits)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        x = z.flatten(1)
        x = self.project(x)
        x = x.view(b, self.base_channels, self.feature_hw, self.feature_hw)
        x = self.tconv_blocks(x)
        x = self.conv(x)
        # ConvTranspose2d with kernel_size=4, stride=1, padding=1 increases
        # spatial size by +1 each block. Pool back to feature_hw so the final
        # FC input dimension remains stable.
        x = F.adaptive_avg_pool2d(x, (self.feature_hw, self.feature_hw))
        x = x.flatten(1)
        x = self.head(x)
        return torch.sigmoid(x)
