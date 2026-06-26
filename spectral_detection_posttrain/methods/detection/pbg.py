import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySpatialBoundaryGate(nn.Module):
    """Joint frequency-spatial boundary gate for ROI features.

    The module extracts boundary cues from two complementary views:

    1. **Spatial view**: a small conv edge detector on the raw ROI feature.
    2. **Frequency view**: phase-only reconstruction via rFFT2 -> magnitude mask
       -> iRFFT2, which suppresses smooth regions and highlights structural
       boundaries.  A configurable magnitude mask prevents unstable low-energy
       frequency bins from contaminating the reconstruction.

    The two boundary maps are fused into a single spatial attention gate and
    applied as a residual: ``x + alpha * tanh(gate) * x``.  ``alpha`` is
    initialised to a small positive value and the fusion conv is initialised to
    zero, so the module starts as an identity mapping.  Using ``tanh`` instead of
    a saturating sigmoid gives the gate a much larger dynamic range near zero and
    avoids the flat-gate problem observed in FSBG v2.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 1e-2,
        phase_mask: str = "soft",
        eps: float = 1e-6,
    ):
        super().__init__()
        if phase_mask not in {"none", "hard", "soft"}:
            raise ValueError(f"Unknown phase_mask: {phase_mask}")
        self.phase_mask = phase_mask
        self.eps = eps

        hid = max(1, channels // 4)
        self.spatial_edge = nn.Sequential(
            nn.Conv2d(channels, hid, 3, padding=1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 1, 3, padding=1, bias=False),
        )
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(channels, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 1, 1, bias=False),
        )
        # FSBG v3: direct sum of spatial + phase edge maps, passed through tanh.
        # Avoids a conv fusion layer that was collapsing the gate to a near-
        # constant value.  Spatial/phase encoders remain learnable.
        self.fusion_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def _phase_boundary(self, x: torch.Tensor) -> torch.Tensor:
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        phase = torch.angle(fr)

        if self.phase_mask == "none":
            phase_only = torch.exp(1j * phase)
        elif self.phase_mask == "hard":
            mask = (mag > self.eps).float()
            phase_only = mask * torch.exp(1j * phase)
        else:  # soft
            # Soft gate: trust phase more where magnitude is large relative to
            # the per-channel mean.  This avoids amplifying noise in low-energy
            # frequency bins.
            mean_mag = mag.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            mask = (mag / mean_mag).clamp(0.0, 1.0)
            phase_only = mask * torch.exp(1j * phase)

        pb = torch.fft.irfft2(phase_only, s=x.shape[-2:], norm="ortho")
        pb = F.relu(pb)
        return self.phase_encoder(pb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_edge(x)
        phase = self._phase_boundary(x)
        gate = torch.tanh(spatial + phase + self.fusion_bias)
        return x + self.alpha * gate * x


# Backward-compatible alias for earlier CLI/config names.
PhaseBoundaryGate = FrequencySpatialBoundaryGate
