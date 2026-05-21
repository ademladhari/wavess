"""Watermark Encoder E (Fig. 3 / Sec. III-C).

Paper description (Sec. III-C, p. 13954):
    "E is composed of four pairs of combination layers, containing a
     convolutional layer, a batch normalization layer, and a ReLU layer.
     The following are two separate linear layers to obtain the mean
     vector and the variance vector, respectively. Finally, zT is derived
     through the reparameterization operation on these two vectors."

Output: watermarked latent zT in R^{C x H x W}, C=4, H=W=64 (Sec. IV-A.4).
Loss (eq. 2): L_d = KL( N(mu, sigma^2) || N(0, 1) ).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WatermarkEncoder(nn.Module):
    """Binary watermark -> Gaussian latent zT (4 x 64 x 64)."""

    def __init__(
        self,
        n_bits: int,
        latent_channels: int = 4,
        latent_size: int = 64,
        base_channels: int = 64,
        conv_blocks: int = 4,
        feature_hw: int = 8,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.base_channels = base_channels
        self.feature_hw = feature_hw

        self.project = nn.Linear(n_bits, base_channels * feature_hw * feature_hw)

        blocks = []
        for _ in range(conv_blocks):
            blocks.append(
                nn.Conv2d(
                    base_channels,
                    base_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            blocks.append(nn.BatchNorm2d(base_channels))
            blocks.append(nn.ReLU(inplace=True))
        self.conv_blocks = nn.Sequential(*blocks)

        latent_dim = latent_channels * latent_size * latent_size
        feat_dim = base_channels * feature_hw * feature_hw
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def encode(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = w.shape[0]
        x = self.project(w)
        x = x.view(b, self.base_channels, self.feature_hw, self.feature_hw)
        x = self.conv_blocks(x)
        x = x.flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(w)
        z = self.reparameterize(mu, logvar)
        b = w.shape[0]
        z_image = z.view(
            b, self.latent_channels, self.latent_size, self.latent_size
        )
        return z_image, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(mu, sigma^2) || N(0,1) ), averaged per-element.

        Per eq. (2) of the paper:
            L_d = -0.5 * (2*log(sigma) + 1 - sigma^2 - mu^2)
        With logvar = 2*log(sigma):
            L_d = -0.5 * (logvar + 1 - exp(logvar) - mu^2)
        """
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()
