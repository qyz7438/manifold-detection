"""Rotation-equivariant / rotation-invariant FFT module for remote sensing.

Remote-sensing objects appear at arbitrary orientations.  This module averages
a feature map over a discrete set of rotations inside the Fourier domain:
for each rotation angle the spatial input is rotated, transformed to the
frequency domain, optionally gated, transformed back to the spatial domain,
rotated back by the inverse angle, and finally pooled across angles.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_tensor(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a batch of feature maps by ``angle`` radians.

    The rotation is performed with ``F.affine_grid`` + ``F.grid_sample`` around
    the image center using bilinear interpolation.  Zero padding is used for
    out-of-bounds regions.

    Args:
        x: tensor of shape ``(B, C, H, W)``.
        angle: rotation angle in radians.

    Returns:
        Rotated tensor of the same shape as ``x``.
    """
    b, _, h, w = x.shape
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    theta = x.new_tensor(
        [
            [cos_a, sin_a, 0.0],
            [-sin_a, cos_a, 0.0],
        ]
    )
    theta = theta.unsqueeze(0).expand(b, -1, -1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


class RotationEquivariantFFT(nn.Module):
    """Rotation-pooled spectral enhancement module.

    The forward pass produces a rotation-stabilized feature map by applying
    the same spectral residual block to several rotated copies of the input,
    rotating the results back to the canonical orientation, and pooling across
    angles.

    Args:
        channels: number of input/output channels.
        n_angles: number of equi-spaced rotations in ``[0, 2*pi)``.
        gate_strength: multiplier for the learnable frequency-domain magnitude
            gate.  A value of ``0.0`` makes the module a pure rotation pool.
        pool: aggregation operation across angles, either ``"mean"`` or
            ``"max"``.
    """

    def __init__(
        self,
        channels: int,
        n_angles: int = 8,
        gate_strength: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()
        if n_angles < 1:
            raise ValueError("n_angles must be a positive integer")
        if pool not in {"mean", "max"}:
            raise ValueError(f"pool must be 'mean' or 'max', got {pool}")

        self.channels = channels
        self.n_angles = n_angles
        self.gate_strength = gate_strength
        self.pool = pool

        angles = torch.linspace(0.0, 2.0 * math.pi, n_angles + 1)[:-1]
        self.register_buffer("angles", angles)

        # A single learnable scalar per channel acts on the Fourier magnitude.
        self.freq_gate = nn.Parameter(torch.zeros(channels))

    def _spectral_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a lightweight magnitude gate in the frequency domain.

        Args:
            x: real-valued feature map of shape ``(B, C, H, W)``.

        Returns:
            Real-valued feature map of the same shape.
        """
        h, w = x.shape[-2:]
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + 1e-3)

        gate = torch.sigmoid(self.freq_gate.view(1, self.channels, 1, 1))
        mag = mag * (1.0 + self.gate_strength * (gate - 0.5))

        fr_mod = mag * torch.exp(1j * pha)
        out = torch.fft.irfft2(fr_mod, s=(h, w), norm="ortho")
        return F.relu(out, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a rotation-pooled spectral feature map.

        Args:
            x: real-valued feature map of shape ``(B, C, H, W)``.

        Returns:
            Rotation-pooled feature map of the same shape as ``x``.
        """
        if x.ndim != 4:
            raise ValueError("RotationEquivariantFFT expects a 4-D input tensor")

        outputs: list[torch.Tensor] = []
        for angle in self.angles:
            angle_value = angle.item()
            x_rot = _rotate_tensor(x, angle_value)
            f_rot = self._spectral_residual(x_rot)
            f_canonical = _rotate_tensor(f_rot, -angle_value)
            outputs.append(f_canonical)

        stacked = torch.stack(outputs, dim=1)  # (B, A, C, H, W)

        if self.pool == "mean":
            return stacked.mean(dim=1)
        # Max pooling is performed on real-valued spatial features.
        return stacked.max(dim=1)[0]
