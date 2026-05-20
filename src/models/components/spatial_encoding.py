import math

import torch
import torch.nn as nn


class SpatialEncoding(nn.Module):
    def __init__(self, fourier_bands: int, fourier_proj_dim: int):
        super().__init__()
        self.fourier_proj_dim = fourier_proj_dim
        freqs = math.pi * (2.0 ** torch.arange(fourier_bands, dtype=torch.float32))
        # freqs are not registered as model parameters, but as buffers to be stored in the state dict and moved to device
        self.register_buffer("freqs", freqs)
        self.fourier_proj = nn.Conv2d(4 * fourier_bands, fourier_proj_dim, kernel_size=1)

    @property
    def extra_channels(self) -> int:
        return self.fourier_proj_dim

    def forward(self, coord_grid: torch.Tensor) -> torch.Tensor:
        # coord_grid: (B, 2, H, W) -- physical coords in meters
        x, y = coord_grid[:, 0:1], coord_grid[:, 1:2]  # (B, 1, H, W) each
        # freqs: (L,) -> (1, L, 1, 1) for broadcasting
        f = self.freqs[None, :, None, None]
        # (B, L, H, W) each
        parts = [torch.sin(f * x), torch.cos(f * x), torch.sin(f * y), torch.cos(f * y)]
        encoded = torch.cat(parts, dim=1)  # (B, 4L, H, W)
        return self.fourier_proj(encoded)  # (B, proj_dim, H, W)
