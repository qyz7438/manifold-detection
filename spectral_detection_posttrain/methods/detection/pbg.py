import torch.nn as nn


class PhaseBoundaryGate(nn.Module):
    """No-op stub for the Phase Boundary Gate."""

    def __init__(self, channels: int, alpha_init: float = 0.0):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x
