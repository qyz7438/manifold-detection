"""Complex-domain ROI feature refinement modules.

These modules operate directly on the complex-valued spectrum obtained from
``torch.fft.rfft2`` without decoupling into magnitude/phase.  The real and
imaginary parts are refined jointly, preserving the coupled frequency/phase
relationship that magnitude+phase decomposition discards.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexROIRefinement(nn.Module):
    """Lightweight complex-domain residual refinement for ROI features.

    Args:
        channels: number of input/output channels.
        hidden: hidden channels for the 1x1 bottleneck.  Defaults to
            ``max(channels // 2, 8)``.
    """

    def __init__(self, channels: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(channels // 2, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, 1, bias=False),
        )
        self.scale = nn.Parameter(torch.zeros(1))

        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_repr = torch.fft.rfft2(x, norm="ortho")
        real_imag = torch.cat([F_repr.real, F_repr.imag], dim=1)
        delta = self.conv(real_imag)
        dr, di = torch.chunk(delta, 2, dim=1)
        F_refined = torch.complex(
            F_repr.real + self.scale * dr,
            F_repr.imag + self.scale * di,
        )
        freq_out = torch.fft.irfft2(F_refined, s=x.shape[-2:], norm="ortho")
        return x + self.scale * F.relu(freq_out, inplace=False)


class ComplexHighDimManifold(nn.Module):
    """Complex-domain ROI refinement with a high-dimensional latent expansion.

    This combines the "complete complex information" and "high-dimensional
    manifold" ideas: the real/imaginary spectrum is expanded to a higher
    dimensional space, transformed non-linearly, and projected back before
    reconstruction.

    Args:
        channels: number of input/output channels.
        expansion: channel expansion factor in the latent space.
    """

    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        mid = max(channels * expansion, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels * 2, 1, bias=False),
        )
        self.scale = nn.Parameter(torch.zeros(1))

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_repr = torch.fft.rfft2(x, norm="ortho")
        real_imag = torch.cat([F_repr.real, F_repr.imag], dim=1)
        delta = self.net(real_imag)
        dr, di = torch.chunk(delta, 2, dim=1)
        F_refined = torch.complex(
            F_repr.real + self.scale * dr,
            F_repr.imag + self.scale * di,
        )
        freq_out = torch.fft.irfft2(F_refined, s=x.shape[-2:], norm="ortho")
        return x + self.scale * F.relu(freq_out, inplace=False)


# Aliases for the build_detector factory.
CRR = ComplexROIRefinement
CHDM = ComplexHighDimManifold
