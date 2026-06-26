"""Spectral-domain image classification head.

Splits the backbone feature map into magnitude and phase components via a 2D
real FFT and processes each branch with global statistics before fusing the
logits with a learnable scalar weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _GlobalStatsPool2d(nn.Module):
    """Global mean and standard deviation pooling over spatial dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the concatenation of channel-wise mean and std.

        Args:
            x: tensor of shape ``(..., C, H, W)``.

        Returns:
            Tensor of shape ``(..., 2*C)``.
        """
        mean = x.mean(dim=(-2, -1))
        std = x.std(dim=(-2, -1), unbiased=False)
        return torch.cat([mean, std], dim=-1)


class _CircularStatsPool2d(nn.Module):
    """Circular mean pooling for angular data such as FFT phases."""

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the concatenation of channel-wise mean sin/cos components.

        Args:
            theta: tensor of shape ``(..., C, H, W)`` containing angles.

        Returns:
            Tensor of shape ``(..., 2*C)``.
        """
        sin = torch.sin(theta).mean(dim=(-2, -1))
        cos = torch.cos(theta).mean(dim=(-2, -1))
        return torch.cat([sin, cos], dim=-1)


class SpectralClassifierHead(nn.Module):
    """Magnitude/phase classification head in the Fourier domain.

    The head applies ``torch.fft.rfft2`` to the input feature map, extracts
    magnitude and phase, pools each with global statistics, and fuses the two
    logit branches using a learnable sigmoid weight.

    Args:
        in_channels: number of channels in the input feature map.
        num_classes: number of output classes.
        hidden_dim: hidden dimension of the two MLP heads.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.mag_pool = _GlobalStatsPool2d()
        self.phase_pool = _CircularStatsPool2d()

        self.mag_head = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
        # Scalar fusion weight initialized so alpha starts near 0.5.
        self.fusion_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fused magnitude/phase logits.

        Args:
            x: input feature map of shape ``(B, C, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        f = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(f)
        phase = torch.angle(f)

        logits_mag = self.mag_head(self.mag_pool(mag))
        logits_phase = self.phase_head(self.phase_pool(phase))

        alpha = torch.sigmoid(self.fusion_weight)
        return alpha * logits_mag + (1 - alpha) * logits_phase

    def forward_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits using only the magnitude branch.

        Args:
            x: input feature map of shape ``(B, C, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        f = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(f)
        return self.mag_head(self.mag_pool(mag))

    def forward_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits using only the phase branch.

        Args:
            x: input feature map of shape ``(B, C, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        f = torch.fft.rfft2(x, norm="ortho")
        phase = torch.angle(f)
        return self.phase_head(self.phase_pool(phase))
