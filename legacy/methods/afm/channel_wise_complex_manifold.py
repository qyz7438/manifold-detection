"""Channel-wise complex spectral manifold refinement for detection ROI features.

This module combines the two ideas raised by the user:

1. **High-dimensional manifold**: each channel's complex spectrum is mapped to a
   latent manifold via ``ComplexSpectralManifold`` (a complex-valued MLP
   autoencoder).
2. **Complete complex-domain information**: the manifold operates directly on
   ``torch.cfloat`` tensors returned by ``torch.fft.rfft2``, preserving both
   magnitude and phase in their coupled form.

The manifold is applied in a channel-wise, weight-sharing fashion so that the
per-channel input dimension stays small (e.g. ``7 * 4 = 28`` for a 7x7 ROI
feature), keeping the module lightweight.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)


class ChannelWiseComplexSpectralManifold(nn.Module):
    """Refine ROI features with a channel-wise complex spectral autoencoder.

    Args:
        channels: number of input/output channels.
        height: spatial height of the ROI feature (also the rfft height).
        width_r: number of columns returned by ``rfft2`` for the above height,
            i.e. ``width // 2 + 1``.
        latent_dim: dimensionality of the complex latent manifold.  Defaults to
            ``height * width_r`` so that an exact identity initialization is used.
    """

    def __init__(
        self,
        channels: int,
        height: int = 7,
        width_r: int = 4,
        latent_dim: int | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width_r = width_r
        in_dim = height * width_r
        self.manifold = ComplexSpectralManifold(
            in_dim=in_dim,
            latent_dim=latent_dim or in_dim,
            hidden_dim=in_dim,
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.height or x.shape[-1] < self.width_r:
            raise ValueError(
                f"Input spatial shape {x.shape[-2:]} does not match "
                f"expected (H={self.height}, W>=width_r={self.width_r})"
            )

        F_repr = torch.fft.rfft2(x, norm="ortho")
        b, c, h, w_r = F_repr.shape

        # Channel-wise flatten: each channel's spectrum becomes a complex vector.
        F_flat = F_repr.reshape(b * c, h * w_r)

        # Encode-decode on the complex manifold.
        F_rec = self.manifold(F_flat)

        # Reshape back to complex feature map.
        F_rec = F_rec.reshape(b, c, h, w_r)

        freq_out = torch.fft.irfft2(F_rec, s=x.shape[-2:], norm="ortho")
        return x + self.scale * F.relu(freq_out, inplace=False)


# Alias for the factory.
CWCSM = ChannelWiseComplexSpectralManifold
