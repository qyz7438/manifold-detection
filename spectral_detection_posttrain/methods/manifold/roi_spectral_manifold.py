"""ROI-level low-rank complex spectral residual adapter.

This module is inserted after RoI Align and before the box head flatten step:

    RoI feature (real, N, C, H, W)
        -> rfft2                  (complex spectrum)
        -> ComplexSpectralAdapter (low-rank per-frequency-position transform)
        -> deltaF = adapter(F) - F
        -> optional (radius, angle) gate
        -> irfft2                 (spatial residual)
        -> x' = x + alpha * delta_x

This is a 7x7 low-resolution counterpart to ``FPNSpectralManifold`` and is
intended mainly as a control to see whether spectral adaptation helps more at
FPN level or at RoI level.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.fpn_spectral_manifold import (
    _make_gate_mlp,
)


def _roi_freq_coords(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return radius and angle maps for an rfft2 spectrum of size (H, W//2+1)."""
    u = torch.fft.fftfreq(height, device=device, dtype=dtype)
    v = torch.fft.rfftfreq(width, device=device, dtype=dtype)
    U, V = torch.meshgrid(u, v, indexing="ij")
    radius = torch.sqrt(U**2 + V**2)
    angle = torch.atan2(U, V)
    rmax = radius.max()
    if rmax > 0:
        radius = radius / rmax
    angle = (angle + math.pi) / (2.0 * math.pi)
    return radius, angle


class ROISpectralManifold(nn.Module):
    """Low-rank complex spectral residual adapter applied to RoI-aligned features.

    Args:
        channels: channel depth of RoI features (e.g. 256).
        latent_dim: latent dimension of the adapter bottleneck.  Defaults to
            ``channels // 4``.
        hidden_dim: hidden width of the encoder/decoder MLPs.
        use_freq_coords: if ``True``, use radius/angle gate for frequency
            selective modulation.
        gate_activation: gate output non-linearity (``sigmoid``, ``sigmoid2``,
            ``tanh``).
        suppress_dc: if ``True``, zero out the DC bin of the residual.
        init_alpha: initial residual mixing strength.
    """

    def __init__(
        self,
        channels: int = 256,
        latent_dim: int | None = None,
        hidden_dim: int | None = None,
        use_freq_coords: bool = True,
        gate_activation: str = "sigmoid",
        suppress_dc: bool = False,
        init_alpha: float = 1e-3,
    ):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim or max(1, channels // 4)
        self.hidden_dim = hidden_dim or max(1, channels // 2)
        self.use_freq_coords = use_freq_coords
        self.gate_activation = gate_activation
        self.suppress_dc = suppress_dc

        self.adapter = ComplexSpectralManifold(
            in_dim=channels,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            near_identity=True,
            near_identity_noise=1e-3,
        )

        # Scalar residual mixing strength.
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        if self.use_freq_coords:
            self.gate = _make_gate_mlp(
                2,
                self.hidden_dim,
                channels,
                activation=gate_activation,
            )

        # Running diagnostics accumulated over one epoch.
        self._stats_buffer: dict[str, list[torch.Tensor]] = {}
        self._last_stats: dict[str, float] = {}

    def reset_stats(self) -> None:
        """Clear the running diagnostic buffer."""
        self._stats_buffer = {}

    def _accumulate_stats(self, stats: dict[str, torch.Tensor]) -> None:
        for key, value in stats.items():
            self._stats_buffer.setdefault(key, []).append(value.detach())

    def epoch_stats(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, values in self._stats_buffer.items():
            if values:
                result[key] = float(torch.stack(values).mean().item())
        self._last_stats = result
        return result

    def last_stats(self) -> dict[str, float]:
        return self._last_stats

    def _make_gate(
        self,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.use_freq_coords:
            return None
        radius, angle = _roi_freq_coords(height, width, device, dtype)
        coords = torch.stack([radius, angle], dim=-1)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        coords = coords.reshape(batch_size * height * coords.size(2), -1)
        return self.gate(coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFT -> low-rank adapter -> iFFT delta to RoI features.

        Args:
            x: real RoI feature tensor of shape ``(N, C, H, W)``.

        Returns:
            Real tensor of the same shape as ``x``.
        """
        if x.ndim != 4:
            raise ValueError(f"ROISpectralManifold expects 4D input, got {x.ndim}D")
        n, c, h, w = x.shape
        if n == 0:
            # No RoIs during inference; skip the FFT to avoid empty-batch issues.
            return x
        if c != self.channels:
            raise ValueError(
                f"Input has {c} channels, expected {self.channels}"
            )

        orig_dtype = x.dtype
        x_f = x.float()

        # 1) Forward FFT.
        F_repr = torch.fft.rfft2(x_f, norm="ortho")
        _, _, h_f, w_r = F_repr.shape

        # 2) Flatten each spatial-frequency position to a C-dim vector.
        F_flat = F_repr.permute(0, 2, 3, 1).reshape(n * h_f * w_r, c)

        # 3) Low-rank complex adapter.
        F_rec = self.adapter(F_flat)

        # 4) Frequency-domain delta.
        F_delta = F_rec - F_flat

        # 5) Optional frequency-coordinate gate.
        if self.use_freq_coords:
            gate = self._make_gate(h, w, n, F_delta.device, F_delta.real.dtype)
            F_delta = F_delta * gate

        # 6) Optionally suppress the DC bin.
        if self.suppress_dc:
            dc_indices = torch.arange(n, device=F_delta.device) * (h_f * w_r)
            F_delta[dc_indices] = 0.0

        # 7) Reshape and inverse FFT.
        F_delta = F_delta.reshape(n, h_f, w_r, c).permute(0, 3, 1, 2)
        spatial_delta = torch.fft.irfft2(F_delta, s=(h, w), norm="ortho")

        # 8) Residual update.
        alpha = self.alpha
        out = x_f + alpha * spatial_delta
        out = out.to(orig_dtype)

        # 9) Diagnostics (detached tensors, accumulated).
        with torch.no_grad():
            x_norm = x_f.norm()
            delta_norm = spatial_delta.norm()
            step_stats: dict[str, torch.Tensor] = {
                "roi_sm/alpha": alpha.detach(),
                "roi_sm/raw_rel_delta": delta_norm / (x_norm + 1e-6),
                "roi_sm/eff_rel_delta": (alpha.abs() * delta_norm) / (x_norm + 1e-6),
                "roi_sm/cos_delta_x": F.cosine_similarity(
                    spatial_delta.reshape(-1), x_f.reshape(-1), dim=0
                ),
                "roi_sm/freq_residual_ratio": (F_rec - F_flat).norm()
                / (F_flat.norm() + 1e-6),
            }
        self._accumulate_stats(step_stats)
        return out
