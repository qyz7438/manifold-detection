r"""Cross-modal low-energy transport for image-text alignment.

The modules in this file bridge real-valued multimodal features (e.g. CLIP
embeddings) with the complex spectral manifold infrastructure from Plan A.
A text feature is interpreted as a displacement field on the image feature
manifold, and the displacement is applied through Chord transport so that the
resulting path is low-energy and structurally smooth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold.chord_transport import (
    ChordTransport,
)
from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)


def _real_to_complex(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """Project a real tensor to a complex tensor via a real linear map.

    The linear layer is assumed to have output dimension ``2 * K``; the first
    half becomes the real part and the second half the imaginary part.

    Args:
        x: real tensor of shape ``(..., in_dim)``.
        layer: real linear layer with output dimension ``2 * K``.

    Returns:
        Complex tensor of shape ``(..., K)``.
    """
    out = layer(x)
    mid = out.shape[-1] // 2
    return torch.complex(out[..., :mid], out[..., mid:])


def _complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """Concatenate real and imaginary parts of a complex tensor.

    Args:
        z: complex tensor of shape ``(..., K)``.

    Returns:
        Real tensor of shape ``(..., 2 * K)``.
    """
    return torch.cat([z.real, z.imag], dim=-1)


class CrossModalTransport(nn.Module):
    r"""Learn a low-energy text-guided displacement on the image feature manifold.

    The text feature is projected to a complex latent displacement. The image
    feature is first lifted to the complex spectral manifold, transported
    along the text-induced direction using :class:`ChordTransport`, and then
    projected back to the original real image feature space.

    Args:
        text_dim: dimensionality of text features.
        image_dim: dimensionality of image features.
        manifold: a :class:`ComplexSpectralManifold` instance from Plan A.
        transport: a :class:`ChordTransport` instance sharing ``manifold``.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        manifold: ComplexSpectralManifold,
        transport: ChordTransport,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.manifold = manifold
        self.transport = transport

        # Project real features to complex spectral / latent coordinates.
        self.image_to_spectral = nn.Linear(image_dim, manifold.in_dim * 2)
        self.text_to_latent = nn.Linear(text_dim, manifold.latent_dim * 2)
        # Project transported complex spectral representation back to real image features.
        self.spectral_to_image = nn.Linear(manifold.in_dim * 2, image_dim)

    def forward(self, text_feature: torch.Tensor, image_feature: torch.Tensor) -> torch.Tensor:
        r"""Refine ``image_feature`` towards the direction implied by ``text_feature``.

        Args:
            text_feature: real tensor of shape ``(B, text_dim)``.
            image_feature: real tensor of shape ``(B, image_dim)``.

        Returns:
            Refined real image feature of shape ``(B, image_dim)``.
        """
        if text_feature.shape[:-1] != image_feature.shape[:-1]:
            raise ValueError("text_feature and image_feature must have the same batch shape")

        # Lift the image feature to complex spectral space.
        F_source = _real_to_complex(image_feature, self.image_to_spectral)
        z_source = self.manifold.encode(F_source)

        # Text feature induces a latent displacement.
        delta_z = _real_to_complex(text_feature, self.text_to_latent)
        z_target = z_source + delta_z
        F_target_obs = self.manifold.decode(z_target)

        # Apply Chord transport for a low-energy step.
        F_pred = self.transport(F_source, F_target_obs)

        # Project back to real image feature space.
        return self.spectral_to_image(_complex_to_real(F_pred))

    def transport_energy(self) -> torch.Tensor | None:
        """Return the energy of the last Chord transport step, if any."""
        return self.transport.transport_energy
