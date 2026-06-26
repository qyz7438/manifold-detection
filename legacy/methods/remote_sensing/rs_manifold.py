"""Remote-sensing extension of the Plan A complex spectral manifold.

This module adds explicit coordinates for scale, orientation and land-cover
class to the latent representation learned by ``ComplexSpectralManifold``.
The extra coordinates are injected as additive embeddings, so the decoder is
unchanged and the base auto-encoder remains an identity at initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)


class RemoteSensingManifold(ComplexSpectralManifold):
    """Spectral manifold with explicit remote-sensing latent coordinates.

    The spectral input is first encoded by the parent ``ComplexSpectralManifold``
    and then shifted by learnable embeddings corresponding to the scale of the
    object, its dominant orientation, and an optional semantic class prior
    (e.g. water, vegetation, building).

    Args:
        in_dim: dimensionality of the input spectral vector.
        latent_dim: dimensionality of the latent manifold.
        n_scales: number of discrete scale bins.
        n_orientations: number of discrete orientation bins.
        n_classes: number of semantic class priors.  ``0`` disables the class
            embedding.
        hidden_dim: hidden width of the encoder/decoder MLPs.  Defaults to
            ``in_dim`` so that an identity initialization is possible.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        n_scales: int,
        n_orientations: int,
        n_classes: int = 0,
        hidden_dim: int | None = None,
    ):
        super().__init__(in_dim, latent_dim, hidden_dim)
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.n_classes = n_classes

        self.scale_embed = (
            nn.Embedding(n_scales, latent_dim) if n_scales > 0 else None
        )
        self.orientation_embed = (
            nn.Embedding(n_orientations, latent_dim)
            if n_orientations > 0
            else None
        )
        self.class_embed = (
            nn.Embedding(n_classes, latent_dim) if n_classes > 0 else None
        )

        # Initialize embeddings to zero so that the module is exactly the
        # parent identity mapping when no coordinate is provided.
        for emb in (self.scale_embed, self.orientation_embed, self.class_embed):
            if emb is not None:
                nn.init.zeros_(emb.weight)

    def encode(
        self,
        F: torch.Tensor,
        scale_idx: torch.Tensor | None = None,
        orientation_idx: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a complex spectral tensor with remote-sensing coordinates.

        Args:
            F: complex spectral tensor of shape ``(..., in_dim)``.
            scale_idx: optional integer tensor of shape ``(...)`` selecting the
                scale bin.
            orientation_idx: optional integer tensor of shape ``(...)``
                selecting the orientation bin.
            class_idx: optional integer tensor of shape ``(...)`` selecting the
                semantic class prior.

        Returns:
            Complex latent coordinate tensor of shape ``(..., latent_dim)``.
        """
        z = super().encode(F)

        if scale_idx is not None and self.scale_embed is not None:
            z = z + self.scale_embed(scale_idx)
        if orientation_idx is not None and self.orientation_embed is not None:
            z = z + self.orientation_embed(orientation_idx)
        if class_idx is not None and self.class_embed is not None:
            z = z + self.class_embed(class_idx)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent coordinates back to spectral space.

        Args:
            z: complex latent tensor of shape ``(..., latent_dim)``.

        Returns:
            Reconstructed complex spectral tensor of shape ``(..., in_dim)``.
        """
        return super().decode(z)

    def forward(
        self,
        F: torch.Tensor,
        scale_idx: torch.Tensor | None = None,
        orientation_idx: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Auto-encode a complex spectral tensor with remote-sensing coordinates.

        Args:
            F: complex spectral tensor of shape ``(..., in_dim)``.
            scale_idx: optional integer tensor of shape ``(...)``.
            orientation_idx: optional integer tensor of shape ``(...)``.
            class_idx: optional integer tensor of shape ``(...)``.

        Returns:
            Reconstructed complex spectral tensor of shape ``(..., in_dim)``.
        """
        z = self.encode(F, scale_idx, orientation_idx, class_idx)
        return self.decode(z)
