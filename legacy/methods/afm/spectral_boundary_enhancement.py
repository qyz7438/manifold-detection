"""Spectral Boundary Enhancement (SBE) for object detection.

SBE is a lightweight, detection-motivated spectral module. Instead of learning
arbitrary magnitude/phase gates, it explicitly measures **phase variation in the
frequency domain** (which corresponds to spatial structure / object boundaries)
and uses it to enhance the magnitude spectrum.  This directly targets the
localization task that object detectors struggle with.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBoundaryEnhancement(nn.Module):
    """Enhance ROI features via phase-gradient magnitude modulation.

    The module has at most two scalar parameters:

    - ``alpha``: learned gain for the phase-gradient enhancement (init 0.0).
    - ``scale``: residual scale (init 0.0).

    It can therefore be trained as a near-parameter-free structural prior.

    Args:
        channels: number of input/output channels (unused, kept for API
            compatibility with the detector factory).
        alpha_init: initial value for the learned enhancement gain.
    """

    def __init__(self, channels: int, alpha_init: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.scale = nn.Parameter(torch.zeros(1))
        self._eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_repr = torch.fft.rfft2(x, norm="ortho")
        phase = torch.angle(F_repr)
        magnitude = torch.abs(F_repr)

        # Robust phase differences along the two frequency axes.  Using
        # angle(F_h * F_{h-1}^*) avoids phase-wrap artefacts.
        dphase_h = torch.angle(
            F_repr[:, :, 1:, :] * F_repr[:, :, :-1, :].conj()
        )
        dphase_w = torch.angle(
            F_repr[:, :, :, 1:] * F_repr[:, :, :, :-1].conj()
        )

        # Pad to restore the original rfft shape.
        dphase_h = F.pad(dphase_h, (0, 0, 0, 1), mode="constant", value=0.0)
        dphase_w = F.pad(dphase_w, (0, 1, 0, 0), mode="constant", value=0.0)

        # Phase-gradient magnitude as a proxy for boundary/structural saliency.
        edge_score = torch.sqrt(dphase_h ** 2 + dphase_w ** 2 + self._eps)

        # Enhance magnitude where the phase spectrum changes rapidly.
        magnitude = magnitude * (1.0 + self.alpha * edge_score)

        F_enhanced = magnitude * torch.exp(1j * phase)
        freq_out = torch.fft.irfft2(
            F_enhanced, s=x.shape[-2:], norm="ortho"
        )

        # Identity-preserving residual.
        return x + self.scale * F.relu(freq_out, inplace=False)


# Alias for the factory.
SBE = SpectralBoundaryEnhancement
