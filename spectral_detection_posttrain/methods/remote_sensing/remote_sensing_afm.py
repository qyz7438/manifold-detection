"""Remote-sensing AFM with explicit multi-scale magnitude/phase processing.

Remote-sensing images contain objects that differ by orders of magnitude in
size and that are often oriented arbitrarily.  ``RemoteSensingAFM`` therefore
operates on a small spatial pyramid, modulates the spectrum of each scale
independently, and fuses the scale-specific residuals with a learned
pixel-wise attention map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaleSpectralBlock(nn.Module):
    """Single-scale FFT residual block used inside ``RemoteSensingAFM``.

    At initialization the convolution weights are zero, so the magnitude and
    phase gates are identities and the block produces approximately the
    original feature map (modulo the ReLU non-linearity).

    Args:
        channels: number of input/output channels.
        gate_strength: scalar multiplier applied to the magnitude gate.
        reduction: channel reduction ratio for the 1x1 gating convolutions.
    """

    def __init__(self, channels: int, gate_strength: float = 0.6, reduction: int = 4):
        super().__init__()
        self.gate_strength = gate_strength
        hidden = max(channels // reduction, 8)

        # Magnitude gate: near-identity at initialization because Tanh(0)=0.
        self.mag_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )

        # Phase residual: near-identity at initialization because Tanh(0)=0.
        self.phase_res = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )

        self._eps = 1e-3

        for module in [self.mag_gate, self.phase_res]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a learnable magnitude/phase modulation and return a residual.

        Args:
            x: real-valued feature map of shape ``(B, C, H, W)``.

        Returns:
            Modulated real-valued feature map of the same shape.
        """
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + self._eps)

        mag_delta = self.mag_gate(torch.log1p(mag))
        mag = mag * (1.0 + self.gate_strength * mag_delta)

        pha_delta = self.phase_res(pha)
        pha = pha + pha_delta

        fr_mod = mag * torch.exp(1j * pha)
        out = torch.fft.irfft2(fr_mod, s=x.shape[-2:], norm="ortho")
        return F.relu(out, inplace=False)


class RemoteSensingAFM(nn.Module):
    """Multi-scale spectral augmentation module for remote-sensing features.

    The module builds a small spatial pyramid from the input feature map,
    applies a scale-specific FFT modulation branch to every scale, and fuses
    the resulting residuals with a learned cross-scale attention map.  The
    final output is ``x + residual`` so that the module is an identity mapping
    when the residual branch is zero.

    Args:
        channels: number of input/output channels.
        scales: list of integer downsampling factors.  ``1`` means the original
            resolution; larger values build coarser pyramid levels.
        gate_strength: magnitude gate multiplier shared across all scales.
        reduction: channel reduction ratio for the per-scale gating networks.
    """

    def __init__(
        self,
        channels: int,
        scales: list[int] | None = None,
        gate_strength: float = 0.6,
        reduction: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.scales = list(scales) if scales is not None else [1, 2, 4]
        if any(s < 1 for s in self.scales):
            raise ValueError("all scale factors must be positive integers")

        self.scale_blocks = nn.ModuleList(
            _ScaleSpectralBlock(channels, gate_strength=gate_strength, reduction=reduction)
            for _ in self.scales
        )

        # Cross-scale attention: one scalar weight map per scale.
        self.scale_attn = nn.Conv2d(
            channels * len(self.scales), len(self.scales), 1, bias=False
        )
        nn.init.zeros_(self.scale_attn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Augment a remote-sensing feature map.

        Args:
            x: real-valued feature map of shape ``(B, C, H, W)``.

        Returns:
            Augmented feature map of the same shape.
        """
        if x.ndim != 4:
            raise ValueError("RemoteSensingAFM expects a 4-D input tensor")
        _, _, h, w = x.shape

        scale_outputs: list[torch.Tensor] = []
        for scale, block in zip(self.scales, self.scale_blocks):
            if scale == 1:
                xs = x
            else:
                xs = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            mod = block(xs)
            if scale != 1:
                mod = F.interpolate(
                    mod, size=(h, w), mode="bilinear", align_corners=False
                )
            scale_outputs.append(mod)

        # Pixel-wise softmax attention over scales.
        concat = torch.cat(scale_outputs, dim=1)  # (B, S*C, H, W)
        attn = F.softmax(self.scale_attn(concat), dim=1)  # (B, S, H, W)

        stacked = torch.stack(scale_outputs, dim=1)  # (B, S, C, H, W)
        attn = attn.unsqueeze(2)  # (B, S, 1, H, W)
        fused = (stacked * attn).sum(dim=1)  # (B, C, H, W)

        return x + fused
