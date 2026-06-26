import math

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


class LearnedSpectralGate(nn.Module):
    """Learned frequency-domain magnitude gate (LSG v1).

    LSG differs from FSBG in that it never reconstructs spatial boundary maps.
    Instead it learns a per-channel frequency response directly on the magnitude
    spectrum:

        x --rFFT2--> X
        |X| --log1p + norm --[radius coord]--> Conv1x1 --tanh--> ΔG
        X' = (1 + ΔG) * X
        x_filt = iRFFT2(X')
        output = x + alpha * (x_filt - x)

    Key properties:
      * Strict identity at init: the last conv weight is zero -> ΔG=0, gate=1,
        and the correction term (x_filt - x) is zero.
      * Optional radius coordinate gives the 1x1 conv explicit frequency-position
        awareness, otherwise it only sees amplitude values.
      * Magnitude is log-compressed and channel-normalized so the gate learns
        relative spectral structure rather than raw energy magnitude.
      * FFT is run in float32 to avoid CUDA half-precision shape restrictions.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 0.1,
        use_radius: bool = True,
    ):
        super().__init__()
        self.use_radius = use_radius
        self.eps = 1e-6
        hid = max(1, channels // 4)

        in_ch = channels + (1 if use_radius else 0)
        self.mag_gate = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, 1, bias=False),
        )
        # Zero-init the last conv -> ΔG=0 at start -> strict identity.
        nn.init.zeros_(self.mag_gate[-1].weight)

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def _radius_map(self, h: int, w_half: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Normalized radial frequency coordinate map (B=1, C=1)."""
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(1, 1, h, 1)
        fx = torch.fft.rfftfreq((w_half - 1) * 2, device=device, dtype=dtype).view(1, 1, 1, w_half)
        r = torch.sqrt(fx ** 2 + fy ** 2)
        r = r / (r.max() + self.eps)
        return r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # FFT in float32 for stability / AMP compatibility.
        x_float = x.float()

        X = torch.fft.rfft2(x_float, norm="ortho")

        # Log-compress and per-sample/channel normalize magnitude.
        mag = torch.log1p(torch.abs(X))
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (
            mag.std(dim=(-2, -1), keepdim=True) + self.eps
        )

        if self.use_radius:
            r = self._radius_map(mag.size(-2), mag.size(-1), mag.device, mag.dtype)
            r = r.expand(mag.size(0), 1, mag.size(-2), mag.size(-1))
            gate_input = torch.cat([mag, r], dim=1)
        else:
            gate_input = mag

        delta = torch.tanh(self.mag_gate(gate_input))
        gate = 1.0 + delta  # identity at init because delta=0

        X_out = gate * X
        x_filt = torch.fft.irfft2(X_out, s=x_float.shape[-2:], norm="ortho")

        # Residual correction: strict identity when alpha=0 or gate=1.
        y = x_float + self.alpha * (x_filt - x_float)
        return y.to(orig_dtype)
