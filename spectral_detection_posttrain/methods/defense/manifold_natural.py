r"""Natural image spectral distribution model for adversarial defense.

The model estimates the mean and covariance of clean training spectra so that
the defense can project anomalous frequencies back onto the natural manifold.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NaturalSpectrumModel(nn.Module):
    r"""Prototype mean and covariance of natural image spectra.

    Given a collection of complex spectra (e.g. from ``torch.fft.rfft2``), the
    model computes the mean complex spectrum and a per-bin covariance.  The
    mean serves as the natural prototype used by ``SpectralChordDefense``.

    Args:
        reg: small ridge added to the diagonal covariance for numerical
            stability.
    """

    def __init__(self, reg: float = 1e-6):
        super().__init__()
        self.reg = reg
        self.register_buffer("mean", None)
        self.register_buffer("cov", None)
        self.register_buffer("_fitted", torch.tensor(False))

    def fit(self, spectra: torch.Tensor) -> "NaturalSpectrumModel":
        r"""Estimate mean and covariance from training spectra.

        Args:
            spectra: complex tensor of shape ``(N, ...)`` where ``N`` is the
                number of training samples.  Statistics are computed across
                the first dimension.

        Returns:
            ``self`` for method chaining.
        """
        if not torch.is_complex(spectra):
            raise ValueError("NaturalSpectrumModel expects complex-valued spectra")
        if spectra.shape[0] < 2:
            raise ValueError("Need at least two spectra to estimate covariance")

        mean = spectra.mean(dim=0)
        # Per-bin variance of complex residuals uses |residual|^2.
        cov = ((spectra - mean).abs().pow(2)).mean(dim=0)
        cov = cov + self.reg

        self.mean = mean
        self.cov = cov
        self._fitted = torch.tensor(True)
        return self

    def prototype(self) -> torch.Tensor:
        """Return the mean natural spectrum.

        Returns:
            Complex tensor with the same shape as a single input spectrum.
        """
        if not self._fitted.item():
            raise RuntimeError("NaturalSpectrumModel must be fit before calling prototype()")
        return self.mean

    def anomaly_score(self, spectra: torch.Tensor) -> torch.Tensor:
        r"""Compute per-bin Mahalanobis-style anomaly score.

        The score is :math:`|F - \mu|^2 / (\sigma^2 + \epsilon)`.

        Args:
            spectra: complex tensor of shape matching ``prototype()`` or with
                an additional leading batch dimension.

        Returns:
            Real tensor of the same shape as ``spectra``.
        """
        if not self._fitted.item():
            raise RuntimeError("NaturalSpectrumModel must be fit before calling anomaly_score()")
        residual = (spectra - self.mean).abs().pow(2)
        return residual / self.cov.clamp_min(self.reg)
