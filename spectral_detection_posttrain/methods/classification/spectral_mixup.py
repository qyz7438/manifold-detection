"""Optimal-transport guided spectral mixup augmentation."""

from __future__ import annotations

import torch

from spectral_detection_posttrain.methods.manifold.sinkhorn_ot import SinkhornOT


class SpectralMixup:
    """Mixup in the 2D-DFT domain using OT alignment.

    The magnitude and phase spectra of two images are aligned with entropic
    optimal transport and then convexly combined.  The label is mixed with the
    same interpolation factor ``lambda``.

    Args:
        alpha: Beta distribution parameter.  ``lambda ~ Beta(alpha, alpha)``.
        eps: entropic regularization for the Sinkhorn solver.
        max_iter: number of Sinkhorn iterations.
        p: power of the Euclidean distance used to build the OT cost.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        eps: float = 0.05,
        max_iter: int = 20,
        p: int = 2,
    ):
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.p = p
        self.sinkhorn = SinkhornOT(eps=eps, max_iter=max_iter, p=p, stable=True)

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply spectral mixup to a pair of images and labels.

        Args:
            x1: first image tensor of shape ``(..., H, W)``.
            x2: second image tensor with the same shape as ``x1``.
            y1: first label.  May be a soft label or class index cast to float.
            y2: second label with the same shape as ``y1``.

        Returns:
            ``(x_mix, y_mix)`` where ``x_mix`` has the same spatial shape as
            ``x1`` and ``y_mix = lambda * y1 + (1 - lambda) * y2``.
        """
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have the same shape")
        if y1.shape != y2.shape:
            raise ValueError("y1 and y2 must have the same shape")

        lam = self._sample_lambda(x1.device)

        f1 = torch.fft.rfft2(x1, norm="ortho")
        f2 = torch.fft.rfft2(x2, norm="ortho")

        mag1, mag2 = torch.abs(f1), torch.abs(f2)
        phase1, phase2 = torch.angle(f1), torch.angle(f2)

        mag_mix = self._ot_interpolate(mag1, mag2, lam)
        phase_mix = self._ot_interpolate(phase1, phase2, lam)
        # Wrap interpolated phase back to the principal branch.
        phase_mix = ((phase_mix + torch.pi) % (2 * torch.pi)) - torch.pi

        f_mix = mag_mix * torch.exp(1j * phase_mix)
        x_mix = torch.fft.irfft2(f_mix, s=x1.shape[-2:], norm="ortho")

        y_mix = lam * y1 + (1 - lam) * y2
        return x_mix, y_mix

    def _sample_lambda(self, device: torch.device) -> torch.Tensor:
        """Sample the mixup coefficient from a symmetric Beta distribution."""
        alpha = torch.tensor(self.alpha, device=device)
        beta_dist = torch.distributions.Beta(alpha, alpha)
        return beta_dist.sample().clamp(0.0, 1.0)

    def _ot_interpolate(
        self, a: torch.Tensor, b: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Align ``b`` to ``a`` with OT and return ``lam*a + (1-lam)*aligned_b``.

        Both inputs are flattened to 1D distributions, an optimal coupling is
        computed with log-space Sinkhorn iterations, and each element of ``a``
        is replaced by the barycentric projection of its transport plan row.
        """
        a_vec = a.reshape(-1)
        b_vec = b.reshape(-1)
        n = a_vec.numel()
        m = b_vec.numel()

        mu = torch.full((n,), 1.0 / n, device=a.device, dtype=a.dtype)
        nu = torch.full((m,), 1.0 / m, device=b.device, dtype=b.dtype)

        cost = SinkhornOT.pairwise_cost(a_vec, b_vec, p=self.p)

        # Log-space Sinkhorn to obtain the optimal coupling.
        log_mu = torch.log(mu + 1e-20)
        log_nu = torch.log(nu + 1e-20)
        log_K = -cost / self.sinkhorn.eps
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        for _ in range(self.sinkhorn.max_iter):
            u = log_mu - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
        log_pi = u.unsqueeze(1) + log_K + v.unsqueeze(0)
        pi = torch.exp(log_pi)

        # Barycentric projection: rows sum to one.
        weights = pi / mu.unsqueeze(1)
        aligned_b = weights @ b_vec

        out_vec = lam * a_vec + (1 - lam) * aligned_b
        return out_vec.reshape(a.shape)
