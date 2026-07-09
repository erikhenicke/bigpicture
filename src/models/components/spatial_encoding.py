"""Fourier positional encoding of per-pixel geographic coordinate grids.

Defines :class:`SpatialEncoding`, an ``nn.Module`` that turns a 2-channel grid of
normalized (x, y) coordinates into a multi-channel Fourier feature map (sin/cos at
geometrically spaced frequencies, projected down with a 1x1 conv). Instantiated in
``train/run_experiment.py`` and consumed by ``DualBranch`` / ``SingleBranchModel``
(see ``models/components/branches.py``), which concatenate its output onto the HR
and/or LR image channels before encoding.
"""
import math

import torch
import torch.nn as nn


class SpatialEncoding(nn.Module):
    """Per-pixel Fourier positional encoding, à la NeRF-style sinusoidal features.

    Maps each pixel's normalized (x, y) coordinate to ``4 * fourier_bands`` sin/cos
    features at geometrically spaced frequencies (``pi * 2**0, pi * 2**1, ...``),
    then projects them to ``fourier_proj_dim`` channels with a 1x1 convolution so
    the encoding can be concatenated onto an image tensor as extra input channels.
    """

    def __init__(self, fourier_bands: int, fourier_proj_dim: int):
        """Initialize the encoding's frequency bank and output projection.

        Args:
            fourier_bands (int): Number of frequency bands ``L``. Each band
                contributes 4 raw features (sin/cos of x and y), so the projection
                takes ``4 * fourier_bands`` input channels.
            fourier_proj_dim (int): Number of output channels after the 1x1 conv
                projection; also exposed via :attr:`extra_channels`.
        """
        super().__init__()
        self.fourier_proj_dim = fourier_proj_dim
        freqs = math.pi * (2.0 ** torch.arange(fourier_bands, dtype=torch.float32))
        # freqs are not registered as model parameters, but as buffers to be stored in the state dict and moved to device
        self.register_buffer("freqs", freqs)
        self.fourier_proj = nn.Conv2d(4 * fourier_bands, fourier_proj_dim, kernel_size=1)

    @property
    def extra_channels(self) -> int:
        """int: Number of channels this encoding adds when concatenated onto an image."""
        return self.fourier_proj_dim

    def forward(self, coord_grid: torch.Tensor) -> torch.Tensor:
        """Encode a per-pixel coordinate grid as projected Fourier features.

        Args:
            coord_grid (torch.Tensor): Shape ``(batch_size, 2, H, W)``, dtype
                float32. Channel 0 is the x coordinate, channel 1 is the y
                coordinate, per pixel (as built by ``FMoWMultiScaleDataset``,
                which normalizes coordinates to roughly ``[-1, 1]``; the inline
                comment below predates that normalization and calls them
                "physical coords in meters").

        Returns:
            torch.Tensor: Shape ``(batch_size, fourier_proj_dim, H, W)``, dtype
            float32 -- the projected sin/cos features, ready to be concatenated
            onto an image tensor along the channel dimension.
        """
        # coord_grid: (B, 2, H, W) -- physical coords in meters
        x, y = coord_grid[:, 0:1], coord_grid[:, 1:2]  # (B, 1, H, W) each
        # freqs: (L,) -> (1, L, 1, 1) for broadcasting
        f = self.freqs[None, :, None, None]
        # (B, L, H, W) each
        parts = [torch.sin(f * x), torch.cos(f * x), torch.sin(f * y), torch.cos(f * y)]
        encoded = torch.cat(parts, dim=1)  # (B, 4L, H, W)
        return self.fourier_proj(encoded)  # (B, proj_dim, H, W)
