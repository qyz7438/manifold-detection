import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseBoundaryGate(nn.Module):
    """Local phase/edge boundary gate for ROI features.

    Learns a low-cost spatial attention map from the ROI feature tensor and
    applies a residual gate: ``x + alpha * sigmoid(edge(x)) * x``.
    ``alpha`` is initialized to 0 so the block starts as an identity mapping.
    """

    def __init__(self, channels: int, alpha_init: float = 0.0):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 4), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 4), 1, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self.edge_conv(x)
        gate = torch.sigmoid(edge)
        return x + self.alpha * gate * x
