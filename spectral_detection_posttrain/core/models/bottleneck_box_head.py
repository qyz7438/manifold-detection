r"""Bottleneck box head for reducing intrinsic dimension after RoIAlign.

Replaces the standard TwoMLPHead path:

    (N, 256, 7, 7) -> flatten -> (N, 12544) -> fc6 -> fc7

with a channel-bottleneck path:

    (N, 256, 7, 7) -> 1x1 conv -> (N, C, 7, 7) -> flatten -> (N, C*49)
    -> fc6 -> fc7

The ambient dimension entering the first fully-connected layer drops from
12544 to C * 49, which directly targets the intrinsic-dimension spike we
observed after MultiScaleRoIAlign.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckTwoMLPHead(nn.Module):
    """Two-MLP box head with a 1x1 channel bottleneck before flattening."""

    def __init__(
        self,
        in_channels: int = 256,
        bottleneck_channels: int = 64,
        representation_size: int = 1024,
        grid_size: int = 7,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.representation_size = representation_size
        self.grid_size = grid_size

        self.bottleneck = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=True,
        )
        flattened_dim = bottleneck_channels * grid_size * grid_size

        self.fc6 = nn.Linear(flattened_dim, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        # Initialize conv and linear layers in a way compatible with detection.
        nn.init.kaiming_uniform_(self.bottleneck.weight, a=1)
        if self.bottleneck.bias is not None:
            nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.kaiming_uniform_(self.fc6.weight, a=1)
        nn.init.constant_(self.fc6.bias, 0)
        nn.init.kaiming_uniform_(self.fc7.weight, a=1)
        nn.init.constant_(self.fc7.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels, H, W)
        x = F.relu(self.bottleneck(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"bottleneck_channels={self.bottleneck_channels}, "
            f"representation_size={self.representation_size}, "
            f"grid_size={self.grid_size})"
        )
