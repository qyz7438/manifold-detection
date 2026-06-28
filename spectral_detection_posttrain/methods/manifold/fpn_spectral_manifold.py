"""FPN-level low-rank complex spectral residual adapter.

The module is inserted between the backbone and the RPN/ROI heads:

    FPN feature maps  (real)
        -> rfft2                  (complex per-level spectrum)
        -> ComplexSpectralAdapter (low-rank per-frequency-position channel transform)
        -> deltaF = adapter(F) - F   (learned low-rank spectral residual)
        -> optional (radius, angle, level) gate
        -> irfft2                 (spatial residual)
        -> x' = x + alpha[key] * delta_x

The adapter is shared across FPN levels, but each level has its own residual
strength ``alpha`` and the frequency-coordinate gate receives a normalized
level index so that it can specialise per scale.

Conceptually this is **not** an autoencoder reconstruction loss.  The
``adapter(F) - F`` term is a low-rank, frequency-conditioned residual whose
only training signal comes from the downstream detection loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)


def _freq_coords(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return radius and angle maps for an rfft2 spectrum of size (H, W//2+1).

    The maps are normalized to [0, 1] and have shape ``(H, W//2+1)``.
    """
    u = torch.fft.fftfreq(height, device=device, dtype=dtype)  # (H,)
    v = torch.fft.rfftfreq(width, device=device, dtype=dtype)  # (W_r,)
    U, V = torch.meshgrid(u, v, indexing="ij")  # (H, W_r)
    radius = torch.sqrt(U**2 + V**2)  # (H, W_r)
    angle = torch.atan2(U, V)         # (H, W_r)

    # Normalize radius to [0, 1] (DC component has radius 0).
    rmax = radius.max()
    if rmax > 0:
        radius = radius / rmax
    # Normalize angle to [0, 1].
    angle = (angle + math.pi) / (2.0 * math.pi)
    return radius, angle


def _make_gate_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    activation: str = "sigmoid",
) -> nn.Module:
    """Build a small real-valued gate MLP.

    Args:
        in_dim: input dimension (2 for radius/angle, 3 if level added).
        hidden_dim: hidden width.
        out_dim: output channels.
        activation: one of ``sigmoid`` (default [0,1]), ``sigmoid2`` ([0,2]),
            or ``tanh`` ([-1,1]).
    """
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    ]
    if activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif activation == "sigmoid2":
        layers.append(_ScaledSigmoid(scale=2.0))
    elif activation == "tanh":
        layers.append(nn.Tanh())
    else:
        raise ValueError(f"Unsupported gate activation: {activation}")
    return nn.Sequential(*layers)


class _ScaledSigmoid(nn.Module):
    """Sigmoid scaled to ``[0, scale]``."""

    def __init__(self, scale: float = 2.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale


class FPNSpectralManifold(nn.Module):
    """Low-rank complex spectral residual adapter on FPN feature pyramids.

    Args:
        channels: channel depth of every FPN level (e.g. 256).
        num_levels: number of FPN levels to process.  Only used to derive the
            default ``level_keys`` when the latter is not provided.
        level_keys: explicit ordered list of FPN keys (e.g.
            ``["0", "1", "2", "pool"]``).  ``alpha`` is bound to these keys so
            that different backbones cannot silently mis-align levels.
        latent_dim: latent dimension of the adapter bottleneck.  Defaults to
            ``channels // 4``.
        hidden_dim: hidden width of the encoder/decoder MLPs.
        use_freq_coords: if ``True``, append radius/angle embeddings and use a
            per-bin gate.
        use_level_coords: if ``True`` (default), also append a normalized level
            index to the gate input so that the gate can specialise per FPN
            level while the complex adapter remains shared.
        gate_activation: gate output non-linearity (``sigmoid``, ``sigmoid2``,
            ``tanh``).
        suppress_dc: if ``True``, force the DC frequency bin of the residual
            to zero so that the module cannot globally shift feature means.
        init_alpha: initial per-level residual mixing strength.  A small
            positive value prevents the dead-lock associated with exact
            identity initialization.
    """

    def __init__(
        self,
        channels: int = 256,
        num_levels: int = 4,
        level_keys: list[str] | None = None,
        latent_dim: int | None = None,
        hidden_dim: int | None = None,
        use_freq_coords: bool = True,
        use_level_coords: bool = True,
        gate_activation: str = "sigmoid",
        suppress_dc: bool = False,
        init_alpha: float = 1e-3,
    ):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.level_keys = level_keys or [str(i) for i in range(num_levels)]
        self.latent_dim = latent_dim or max(1, channels // 4)
        self.hidden_dim = hidden_dim or max(1, channels // 2)
        self.use_freq_coords = use_freq_coords
        self.use_level_coords = use_level_coords and use_freq_coords
        self.gate_activation = gate_activation
        self.suppress_dc = suppress_dc

        # Shared low-rank complex adapter across levels.
        self.adapter = ComplexSpectralManifold(
            in_dim=channels,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            near_identity=True,
            near_identity_noise=1e-3,
        )

        # Per-level residual strength, explicitly bound to FPN keys.
        self.alpha = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
            for key in self.level_keys
        })

        # Optional frequency-coordinate gate.
        if self.use_freq_coords:
            gate_in_dim = 2 + (1 if self.use_level_coords else 0)
            self.gate = _make_gate_mlp(
                gate_in_dim,
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
        """Append a single-step tensor stats dict to the running buffer."""
        for key, value in stats.items():
            self._stats_buffer.setdefault(key, []).append(value.detach())

    def epoch_stats(self) -> dict[str, float]:
        """Return the mean of all accumulated diagnostic tensors."""
        result: dict[str, float] = {}
        for key, values in self._stats_buffer.items():
            if values:
                stacked = torch.stack(values)
                result[key] = float(stacked.mean().item())
        self._last_stats = result
        return result

    def last_stats(self) -> dict[str, float]:
        """Return the most recently computed epoch stats (or empty dict)."""
        return self._last_stats

    def _make_gate(
        self,
        height: int,
        width: int,
        batch_size: int,
        level_idx: int,
        num_levels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Build a per-frequency-bin real-valued gate of shape (B*H*W_r, C)."""
        if not self.use_freq_coords:
            return None
        radius, angle = _freq_coords(height, width, device, dtype)
        coords = [radius, angle]
        if self.use_level_coords:
            level_coord = torch.full_like(radius, level_idx / max(1, num_levels - 1))
            coords.append(level_coord)
        coords = torch.stack(coords, dim=-1)  # (H, W_r, K)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        coords = coords.reshape(batch_size * height * coords.size(2), -1)
        return self.gate(coords)  # (B*H*W_r, C)

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply FFT -> low-rank adapter -> iFFT delta to each FPN level.

        Args:
            features: mapping from level key to real feature tensor of shape
                ``(B, C, H, W)``.

        Returns:
            Dictionary with the same keys and spatial shapes as ``features``.
        """
        # Defensive: every configured level must be present.
        missing = set(self.level_keys) - set(features.keys())
        if missing:
            raise KeyError(
                f"FPNSpectralManifold configured levels {self.level_keys} "
                f"missing from input keys {list(features.keys())}; missing={missing}"
            )

        out: dict[str, torch.Tensor] = {}
        step_stats: dict[str, torch.Tensor] = {}
        num_levels = len(self.level_keys)

        for level_idx, key in enumerate(self.level_keys):
            feat = features[key]
            b, c, h, w = feat.shape
            assert c == self.channels, (
                f"FPN level {key} has {c} channels, expected {self.channels}"
            )

            # Work in float32 for FFT stability, especially under AMP.
            orig_dtype = feat.dtype
            feat_f = feat.float()

            # 1) Forward FFT.
            F_repr = torch.fft.rfft2(feat_f, norm="ortho")
            _, _, h_f, w_r = F_repr.shape

            # 2) Flatten each spatial-frequency position to a C-dim vector.
            F_flat = F_repr.permute(0, 2, 3, 1).reshape(b * h_f * w_r, c)

            # 3) Low-rank complex adapter.
            F_rec = self.adapter(F_flat)

            # 4) Delta in the frequency domain (the learned spectral residual).
            F_delta = F_rec - F_flat

            # 5) Optional frequency-coordinate gate.
            if self.use_freq_coords:
                gate = self._make_gate(
                    h, w, b, level_idx, num_levels,
                    F_delta.device, F_delta.real.dtype,
                )
                F_delta = F_delta * gate

            # 6) Optionally suppress the DC bin to avoid global mean shifts.
            if self.suppress_dc:
                # Flatten order: each batch slice starts with the DC bin.
                dc_indices = torch.arange(b, device=F_delta.device) * (h_f * w_r)
                F_delta[dc_indices] = 0.0

            # 7) Reshape and inverse FFT to obtain a spatial residual.
            F_delta = F_delta.reshape(b, h_f, w_r, c).permute(0, 3, 1, 2)
            spatial_delta = torch.fft.irfft2(F_delta, s=(h, w), norm="ortho")

            # 8) Residual update with per-level alpha.
            alpha = self.alpha[key]
            out[key] = feat_f + alpha * spatial_delta
            out[key] = out[key].to(orig_dtype)

            # 9) Diagnostics (detached tensors, no .item() in hot path).
            with torch.no_grad():
                feat_norm = feat_f.norm()
                delta_norm = spatial_delta.norm()
                raw_rel_delta = delta_norm / (feat_norm + 1e-6)
                eff_rel_delta = (alpha.abs() * delta_norm) / (feat_norm + 1e-6)
                cos = F.cosine_similarity(
                    spatial_delta.reshape(-1),
                    feat_f.reshape(-1),
                    dim=0,
                )
                freq_residual_ratio = (F_rec - F_flat).norm() / (F_flat.norm() + 1e-6)
                prefix = f"fpn_sm/{key}"
                step_stats[f"{prefix}_alpha"] = alpha.detach()
                step_stats[f"{prefix}_raw_rel_delta"] = raw_rel_delta
                step_stats[f"{prefix}_eff_rel_delta"] = eff_rel_delta
                step_stats[f"{prefix}_cos_delta_x"] = cos
                step_stats[f"{prefix}_freq_residual_ratio"] = freq_residual_ratio

        self._accumulate_stats(step_stats)
        return out
