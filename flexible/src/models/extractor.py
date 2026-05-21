"""Watermark Extractor Dext (Fig. 3 / Sec. III-D).

Paper description (Sec. III-D, p. 13955):
    "Along with a convolutional layer and a linear layer, Dext includes a
     Transformer encoder with two layers and one head, an MLP module
     consisting of a linear layer, a ReLU layer, and a linear layer, and
     finally a convolution layer."

Input: watermarked image Iw in R^{3 x 512 x 512}.
Output: recovered watermarked latent z'T in R^{4 x 64 x 64}.

Loss (eq. 5): L_ext = MSE(z'T, zT).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WatermarkExtractor(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        in_channels: int = 3,
        latent_channels: int = 4,
        latent_size: int = 64,
        in_downsample: int = 16,
        token_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 1,
        mlp_hidden: int = 256,
        ffn_dim: int = 256,
    ) -> None:
        super().__init__()
        assert image_size % in_downsample == 0
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        self.token_grid = image_size // in_downsample
        self.token_dim = token_dim

        self.patch_conv = nn.Conv2d(
            in_channels,
            token_dim,
            kernel_size=in_downsample,
            stride=in_downsample,
        )
        self.patch_proj = nn.Linear(token_dim, token_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=ffn_dim,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, token_dim),
        )

        # token_grid -> latent_size. For image_size=512 / in_downsample=16 we
        # have token_grid=32 and latent_size=64, so upsample x2 via a single
        # ConvTranspose2d.
        up_stride = latent_size // self.token_grid
        if up_stride == 1:
            self.out_conv = nn.Conv2d(token_dim, latent_channels, kernel_size=3, padding=1)
        else:
            self.out_conv = nn.ConvTranspose2d(
                token_dim,
                latent_channels,
                kernel_size=up_stride * 2,
                stride=up_stride,
                padding=up_stride // 2,
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        x = self.patch_conv(image)
        g = x.shape[-1]
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.patch_proj(tokens)
        tokens = self.transformer(tokens)
        tokens = self.mlp(tokens)
        x = tokens.transpose(1, 2).contiguous().view(b, self.token_dim, g, g)
        z = self.out_conv(x)
        if z.shape[-1] != self.latent_size:
            z = nn.functional.interpolate(
                z, size=(self.latent_size, self.latent_size), mode="bilinear", align_corners=False
            )
        return z
