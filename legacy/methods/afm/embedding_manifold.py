"""Embedding-level manifold refinement for object detection.

This module operates on the high-level representation produced by the box head,
i.e. the vector that the classifier / regressor sees.  It models the manifold of
object-instance embeddings and refines the representation before the final
prediction layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BoxEmbeddingManifold(nn.Module):
    """Lightweight autoencoder-style manifold refinement on box-head embeddings.

    The module is initialized so that ``forward(z) \approx z`` at start, allowing
    it to be inserted before the box predictor without disturbing a pretrained
    detector.

    Args:
        in_features: dimensionality of the box-head output vector.
        hidden_features: bottleneck width.  Defaults to ``in_features // 2``.
        latent_features: manifold dimension.  Defaults to ``in_features // 4``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        latent_features: int | None = None,
    ):
        super().__init__()
        hidden = hidden_features or max(in_features // 2, 64)
        latent = latent_features or max(in_features // 4, 32)

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_features),
        )
        self.scale = nn.Parameter(torch.zeros(1))

        # Initialize the last layer of decoder to zero so that the residual is
        # zero at start.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        residual = self.decoder(self.encoder(z))
        return z + self.scale * residual
