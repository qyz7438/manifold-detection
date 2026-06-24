r"""Manifold-aware AFM blocks for semantic segmentation.

This module extends the MPLSeg magnitude/phase decomposition to a learnable
complex spectral manifold.  High-channel feature maps are split into channel
groups; each spatial-frequency coefficient vector is embedded onto a
low-dimensional manifold, modulated, refined by ChordEdit-style low-energy
transport, and mapped back to the pixel domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_detection_posttrain.methods.manifold.complex_manifold import (
    ComplexSpectralManifold,
)
from spectral_detection_posttrain.methods.manifold.riemannian_metric import (
    AdaptiveRiemannianMetric,
)
from spectral_detection_posttrain.methods.manifold.chord_transport import (
    ChordTransport,
)


class ManifoldAFMBlock(nn.Module):
    r"""Channel-grouped AFM block operating on a learnable complex manifold.

    The block factorises a feature map :math:`x \in \mathbb{R}^{B \times C
    \times H \times W}` into :math:`G = C / D` channel groups of dimension
    :math:`D`.  Each spatial-frequency coefficient of a group is treated as a
    point in :math:`\mathbb{C}^{D}`, embedded onto a manifold, modulated in
    magnitude/phase, refined by Chord transport, and decoded back.

    At initialization the manifold autoencoder is an approximate identity, the
    magnitude/phase networks output zero, and ``residual_scale == 1``.  Hence
    ``forward(x) \approx x`` at the start of training, which keeps the
    segmentation backbone stable.

    Args:
        channels: number of input channels :math:`C`. Must be divisible by
            ``latent_dim`` (or by ``groups`` when provided).
        latent_dim: per-group manifold dimension :math:`D`. Defaults to 32.
        gate_strength: multiplier applied to the magnitude/phase deltas.
        groups: number of channel groups. If ``None``, computed as
            ``channels // latent_dim``.
        use_chord_transport: whether to apply ChordEdit low-energy transport
            after magnitude/phase modulation.
        delta: damping factor used by ``ChordTransport``.
        lambda_step: step size used by ``ChordTransport``.
        hidden_dim: hidden width of the magnitude/phase MLPs. Defaults to
            ``max(latent_dim // 2, 8)``.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    """

    def __init__(
        self,
        channels: int,
        latent_dim: int = 32,
        gate_strength: float = 0.6,
        groups: int | None = None,
        use_chord_transport: bool = True,
        delta: float = 0.15,
        lambda_step: float = 1.0,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        if groups is None:
            groups = channels // latent_dim
        if groups <= 0:
            raise ValueError("groups must be positive")
        if channels % groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by groups ({groups})"
            )

        self.channels = channels
        self.latent_dim = channels // groups
        self.gate_strength = gate_strength
        self.groups = groups
        self.use_chord_transport = use_chord_transport

        # Shared manifold/transport across groups for parameter efficiency.
        self.manifold = ComplexSpectralManifold(
            in_dim=self.latent_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.latent_dim,
        )
        metric = AdaptiveRiemannianMetric(latent_dim=self.latent_dim)
        self.chord_transport = ChordTransport(
            self.manifold,
            metric,
            delta=delta,
            lambda_step=lambda_step,
        )

        # Magnitude and phase modulation networks operate on real latent
        # coordinates.  They are initialized to zero so that the block is
        # approximately identity at start.
        hid = hidden_dim or max(self.latent_dim // 2, 8)
        self.mag_gate = nn.Sequential(
            nn.Linear(self.latent_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, self.latent_dim),
        )
        self.phase_res = nn.Sequential(
            nn.Linear(self.latent_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, self.latent_dim),
        )
        self._zero_init_gates()

        # Must be 1.0 at init to avoid the topology dead-lock described in
        # Plan B: residual_scale=0 blocks gradients through the FFT path.
        self.residual_scale = nn.Parameter(torch.ones(1))

    def _zero_init_gates(self) -> None:
        """Zero-initialize gate networks so that modulation starts as identity."""
        for module in [self.mag_gate, self.phase_res]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _reshape_to_grouped_latent(
        self, F: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Reshape a complex spectrum to a grouped latent tensor.

        Args:
            F: complex tensor of shape ``(B, C, H, W_freq)``.

        Returns:
            ``(F_flat, shape_meta)`` where ``F_flat`` has shape
            ``(B * H * W_freq * groups, latent_dim)`` and ``shape_meta`` stores
            the original ``(B, C, H, W_freq)`` shape.
        """
        B, C, H, Wf = F.shape
        G = self.groups
        D = self.latent_dim
        # (B, G, D, H, Wf) -> (B, G, H, Wf, D) -> (B*G*H*Wf, D)
        F_grouped = F.reshape(B, G, D, H, Wf).permute(0, 1, 3, 4, 2)
        F_flat = F_grouped.reshape(-1, D)
        return F_flat, (B, C, H, Wf)

    def _reshape_to_spectrum(
        self, F_flat: torch.Tensor, shape_meta: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Inverse of ``_reshape_to_grouped_latent``."""
        B, C, H, Wf = shape_meta
        G = self.groups
        D = self.latent_dim
        F_grouped = F_flat.reshape(B, G, H, Wf, D)
        F = F_grouped.permute(0, 1, 4, 2, 3).reshape(B, C, H, Wf)
        return F

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply manifold AFM to a real-valued feature map.

        Args:
            x: tensor of shape ``(B, C, H, W)``.

        Returns:
            Refined feature map of the same shape.
        """
        if x.shape[1] != self.channels:
            raise ValueError(
                f"input has {x.shape[1]} channels but block expects {self.channels}"
            )

        # 1. Forward FFT.
        F = torch.fft.rfft2(x, norm="ortho")

        # 2. Reshape to grouped latent vectors.
        F_flat, shape_meta = self._reshape_to_grouped_latent(F)

        # 3. Embed onto the manifold.
        z = self.manifold.encode(F_flat)

        # 4. Magnitude/phase decoupled modulation in latent space.
        rho, theta = ComplexSpectralManifold.split_magnitude_phase(z)
        mag_delta = self.mag_gate(rho)
        phase_delta = self.phase_res(theta)
        rho_new = rho * (1.0 + self.gate_strength * torch.tanh(mag_delta))
        theta_new = theta + self.gate_strength * torch.tanh(phase_delta)
        z_new = ComplexSpectralManifold.combine_magnitude_phase(rho_new, theta_new)

        # 5. Decode the modulated spectrum.
        F_new_flat = self.manifold.decode(z_new)

        # 6. Chord transport refinement in spectral space.
        if self.use_chord_transport:
            F_refined_flat = self.chord_transport(F_new_flat, F_flat)
        else:
            F_refined_flat = F_new_flat

        # 7. Reshape and inverse FFT.
        F_refined = self._reshape_to_spectrum(F_refined_flat, shape_meta)
        freq_out = torch.fft.irfft2(F_refined, s=x.shape[-2:], norm="ortho")

        # 8. Identity-preserving residual connection.
        return x + self.residual_scale * (freq_out - x)


class ManifoldAFMStack(nn.Module):
    r"""Stack of ``ManifoldAFMBlock`` modules for multi-level feature maps.

    Args:
        channels: list of channel depths, one per level.
        latent_dim: per-group manifold dimension shared across levels.
        gate_strength: shared gate strength, or a list of per-level strengths.
        kwargs: extra arguments forwarded to ``ManifoldAFMBlock``.
    """

    def __init__(
        self,
        channels: list[int],
        latent_dim: int = 32,
        gate_strength: float | list[float] = 0.6,
        **kwargs,
    ):
        super().__init__()
        strengths = (
            [gate_strength] * len(channels)
            if isinstance(gate_strength, (int, float))
            else gate_strength
        )
        if len(strengths) != len(channels):
            raise ValueError(
                f"gate_strength length ({len(strengths)}) does not match "
                f"channels length ({len(channels)})"
            )
        self.blocks = nn.ModuleDict({
            str(i): ManifoldAFMBlock(
                channels=c, latent_dim=latent_dim, gate_strength=s, **kwargs
            )
            for i, (c, s) in enumerate(zip(channels, strengths))
        })

    def forward(self, feature_map: torch.Tensor, level: int) -> torch.Tensor:
        """Apply the level-specific block.

        Args:
            feature_map: tensor of shape ``(B, C, H, W)``.
            level: index into ``channels``.

        Returns:
            Refined feature map of the same shape.
        """
        return self.blocks[str(level)](feature_map)
