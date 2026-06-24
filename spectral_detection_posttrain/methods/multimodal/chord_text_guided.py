r"""ChordEdit-style text-guided image editing/retrieval refinement.

This module extends Plan A's Chord transport from "text edit -> image" to
"text guide -> image feature refinement".  Source and target text captions are
encoded into a real drift vector; the drift is lifted to the complex latent
manifold and applied to the source image feature through low-energy Chord
transport.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold.chord_transport import (
    ChordTransport,
)
from spectral_detection_posttrain.methods.multimodal.cross_modal_transport import (
    _complex_to_real,
    _real_to_complex,
)


class ChordTextGuidedEdit(nn.Module):
    r"""Text-guided image feature editor based on Chord transport.

    Given a source image and a pair of source/target text prompts, the module
    computes the text-induced drift, lifts the image feature onto the spectral
    manifold, applies a low-energy Chord transport step, and maps the result
    back to the original feature space.

    Args:
        image_encoder: callable mapping images to real features of shape
            ``(B, feature_dim)``. May be a ``nn.Module`` or any callable.
        text_encoder: callable mapping text token tensors (or raw text
            indices) to real features of shape ``(B, feature_dim)``.
        transport: a :class:`ChordTransport` instance from Plan A.
        feature_dim: dimensionality of the image and text features produced by
            the encoders.
    """

    def __init__(
        self,
        image_encoder: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        text_encoder: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        transport: ChordTransport,
        feature_dim: int,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.transport = transport
        self.feature_dim = feature_dim

        manifold = transport.manifold
        self.image_to_spectral = nn.Linear(feature_dim, manifold.in_dim * 2)
        self.text_to_latent = nn.Linear(feature_dim, manifold.latent_dim * 2)
        self.spectral_to_feature = nn.Linear(manifold.in_dim * 2, feature_dim)

    def forward(
        self,
        x_source: torch.Tensor,
        text_source: torch.Tensor,
        text_target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Perform a text-guided low-energy edit of ``x_source``.

        Args:
            x_source: source image tensor accepted by ``image_encoder``.
            text_source: source text tensor accepted by ``text_encoder``.
            text_target: target text tensor accepted by ``text_encoder``.

        Returns:
            Edited real feature tensor of shape ``(B, feature_dim)``.  To
            obtain a full image, feed the result into an external decoder.
        """
        # Encode modalities into a shared real feature space.
        f_src = self.image_encoder(x_source)
        v_src = self.text_encoder(text_source)
        v_tar = self.text_encoder(text_target)

        if f_src.shape != v_src.shape or f_src.shape != v_tar.shape:
            raise ValueError("image and text encoders must produce features of the same shape")

        # Lift image feature to complex spectral space.
        F_source = _real_to_complex(f_src, self.image_to_spectral)
        z_source = self.transport.manifold.encode(F_source)

        # Build the text-induced drift in latent space.
        text_drift = v_tar - v_src
        delta_z = _real_to_complex(text_drift, self.text_to_latent)
        z_target = z_source + delta_z
        F_target_obs = self.transport.manifold.decode(z_target)

        # Low-energy Chord transport.
        F_pred = self.transport(F_source, F_target_obs)

        return self.spectral_to_feature(_complex_to_real(F_pred))
