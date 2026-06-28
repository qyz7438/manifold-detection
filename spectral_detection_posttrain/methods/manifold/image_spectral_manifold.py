"""Image-level FFT + complex spectral manifold analysis.

This module is inserted before the backbone:

    input image (real, B, C, H, W)
        -> rfft2       (complex spectrum)
        -> ComplexSpectralManifold (per-frequency-position channel-wise autoencoder)
        -> irfft2      (back to real spatial image)
        -> residual add

The manifold treats each spatial-frequency position as a sample and the
channel dimension (typically RGB=3) as the feature vector.  This is the
"spectral analysis on the original image" counterpart to the ROI-level
manifold that operates on data-feature dimensions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)


class ImageSpectralManifold(nn.Module):
    """FFT + complex spectral manifold applied directly to the input image.

    Args:
        channels: number of input channels (typically 3 for RGB).
        latent_dim: latent dimension of the manifold autoencoder.  Defaults to
            ``channels`` so the module starts as an approximate identity mapping.
    """

    def __init__(
        self,
        channels: int = 3,
        latent_dim: int | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim or channels

        self.manifold = ComplexSpectralManifold(
            in_dim=channels,
            latent_dim=self.latent_dim,
            hidden_dim=channels,
        )
        # Scalar residual mixing strength, initialized to zero (identity).
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFT -> manifold -> iFFT to the input image.

        Args:
            x: real input tensor of shape ``(B, C, H, W)``.

        Returns:
            Real tensor of the same shape as ``x``.
        """
        if x.ndim != 4:
            raise ValueError(f"ImageSpectralManifold expects 4D input, got {x.ndim}D")
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"Input has {c} channels, expected {self.channels}"
            )

        # 1) Forward FFT: (B, C, H, W) -> (B, C, H, W//2+1) complex.
        F_repr = torch.fft.rfft2(x, norm="ortho")
        b_f, c_f, h_f, w_r = F_repr.shape
        assert c_f == c and b_f == b and h_f == h

        # 2) Treat each spatial-frequency position as a C-dim complex vector.
        F_flat = F_repr.permute(0, 2, 3, 1).reshape(b * h * w_r, c)

        # 3) Manifold autoencoder in the complex domain.
        F_rec = self.manifold(F_flat)

        # 4) Reshape back to a complex spectrum.
        F_rec = F_rec.reshape(b, h, w_r, c).permute(0, 3, 1, 2)

        # 5) Inverse FFT back to the original spatial size.
        freq_out = torch.fft.irfft2(F_rec, s=(h, w), norm="ortho")

        # 6) Identity-preserving residual update.
        return x + self.scale * freq_out
