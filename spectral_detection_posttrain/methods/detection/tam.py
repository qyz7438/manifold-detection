import torch.nn as nn


class TaskAlignedManifold(nn.Module):
    """No-op stub for the Task-Aligned Manifold."""

    def __init__(self, in_features: int, latent_dim: int = 256):
        super().__init__()
        self.in_features = in_features

    def forward(self, z):
        return z
