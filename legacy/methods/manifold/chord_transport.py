r"""ChordEdit-style low-energy transport on the spectral manifold.

The transport encodes source and target observations onto the learned
manifold, builds a smoothed control field from their latent residual, and
decodes the transported latent coordinate back to spectral space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.riemannian_metric import (
    AdaptiveRiemannianMetric,
)


class ChordTransport(nn.Module):
    r"""Single-step Chord transport between two complex spectral states.

    Args:
        manifold: a ``ComplexSpectralManifold`` instance.
        metric: an ``AdaptiveRiemannianMetric`` instance (kept for API
            compatibility with future multi-step transport).
        delta: smoothing window in :math:`[0, 1]`. Larger values damp the
            control field more aggressively.
        lambda_step: transport step size.
    """

    def __init__(
        self,
        manifold: ComplexSpectralManifold,
        metric: AdaptiveRiemannianMetric,
        delta: float = 0.15,
        lambda_step: float = 1.0,
    ):
        super().__init__()
        self.manifold = manifold
        self.metric = metric
        if not 0.0 <= delta <= 1.0:
            raise ValueError("delta must be in [0, 1]")
        self.delta = delta
        self.lambda_step = lambda_step
        self._transport_energy: torch.Tensor | None = None

    def forward(self, F_source: torch.Tensor, F_target_obs: torch.Tensor) -> torch.Tensor:
        r"""Transport ``F_source`` towards ``F_target_obs``.

        The control field is a time-weighted average of the source residual
        :math:`R_{src} = z_{tar} - z_{src}` and the target residual
        :math:`R_{tar} = 0` (the observation is already at the target state):

        .. math::
            \hat{u} = \frac{R_{src} + \delta R_{tar}}{1 + \delta}
                    = \frac{R_{src}}{1 + \delta}.

        The latent prediction is :math:`z_{pred} = z_{src} + \lambda \hat{u}`
        and is decoded back to spectral space.

        Args:
            F_source: source complex spectrum, shape ``(..., in_dim)``.
            F_target_obs: target observed complex spectrum, same shape.

        Returns:
            Predicted complex spectrum ``F_pred`` of the same shape.
        """
        if not (torch.is_complex(F_source) and torch.is_complex(F_target_obs)):
            raise ValueError("ChordTransport expects complex-valued spectral inputs")

        z_src = self.manifold.encode(F_source)
        z_tar = self.manifold.encode(F_target_obs)

        R_src = z_tar - z_src
        R_tar = torch.zeros_like(R_src)  # residual vanishes at the observation.
        u_hat = (R_src + self.delta * R_tar) / (1.0 + self.delta)

        self._transport_energy = (u_hat.abs() ** 2).sum(dim=-1)

        z_pred = z_src + self.lambda_step * u_hat
        F_pred = self.manifold.decode(z_pred)
        return F_pred

    @property
    def transport_energy(self) -> torch.Tensor | None:
        """Squared :math:`L^2` norm of the last control field.

        Returns ``None`` before ``forward`` has been called in the current
        session.
        """
        return self._transport_energy
