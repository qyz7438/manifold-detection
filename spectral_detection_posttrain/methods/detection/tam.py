import torch
import torch.nn as nn


class TaskAlignedManifold(nn.Module):
    """Task-aligned manifold refinement on box-head embeddings.

    Encodes the penultimate embedding to a compact latent space and decodes a
    residual that is added back with a learnable scale.  The scale is initialized
    to zero so the block starts as an identity mapping.
    """

    def __init__(self, in_features: int, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, in_features),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        # Zero-initialize decoder output so the residual starts at zero.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(z)
        residual = self.decoder(latent)
        return z + self.scale * residual
