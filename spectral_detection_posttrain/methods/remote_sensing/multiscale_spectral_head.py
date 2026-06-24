"""Multi-scale spectral detection head for remote-sensing FPN features.

This head applies a remote-sensing AFM to every FPN level independently and
projects the augmented features to a common channel depth.  It can be inserted
before the final classification / regression subnets of a detector without
changing the downstream architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.remote_sensing.remote_sensing_afm import (
    RemoteSensingAFM,
)


class MultiScaleSpectralHead(nn.Module):
    """Per-level spectral head with optional cross-scale channel projection.

    Args:
        in_channels: list of channel depths, one per FPN level.
        out_channels: target channel depth for each output level.  If ``None``,
            the original channel depth of each level is preserved.
        scales: scale factors passed to each ``RemoteSensingAFM``.
        gate_strength: magnitude gate multiplier passed to each
            ``RemoteSensingAFM``.
        reduction: channel reduction ratio passed to each ``RemoteSensingAFM``.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int | None = None,
        scales: list[int] | None = None,
        gate_strength: float = 0.6,
        reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels

        self.afm_blocks = nn.ModuleList(
            RemoteSensingAFM(
                channels=c,
                scales=scales,
                gate_strength=gate_strength,
                reduction=reduction,
            )
            for c in self.in_channels
        )

        if out_channels is not None:
            self.projections = nn.ModuleList(
                nn.Conv2d(c, out_channels, 1, bias=False) for c in self.in_channels
            )
        else:
            self.projections = None

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply spectral augmentation to a list of FPN feature maps.

        Args:
            features: list of real-valued feature maps.  Each tensor has shape
                ``(B, C_i, H_i, W_i)`` where ``C_i`` matches ``in_channels[i]``.

        Returns:
            List of augmented feature maps.  If ``out_channels`` was specified,
            every output tensor has ``out_channels`` channels; otherwise the
            original channel depths are preserved.
        """
        if len(features) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature maps, got {len(features)}"
            )

        outputs: list[torch.Tensor] = []
        for i, x in enumerate(features):
            y = self.afm_blocks[i](x)
            if self.projections is not None:
                y = self.projections[i](y)
            outputs.append(y)
        return outputs
