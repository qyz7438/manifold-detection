from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OldAFMBlock(nn.Module):
    """Round 2.6/3.1 non-identity AFM kept only for controlled diagnostics."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mag_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.phase_res = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )
        self._eps = 1e-3
        for module in [self.mag_gate, self.phase_res]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_repr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(f_repr)
        pha = torch.angle(f_repr + self._eps)
        mag = mag * (1.0 - self.mag_gate(torch.log1p(mag)))
        pha = pha + self.phase_res(pha)
        f_mod = mag * torch.exp(1j * pha)
        out = torch.fft.irfft2(f_mod, s=x.shape[-2:], norm="ortho")
        return F.relu(out, inplace=True)


class AFMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, residual_mode: str = "current"):
        super().__init__()
        if residual_mode not in {"current", "delta", "norm_delta"}:
            raise ValueError(f"Unknown AFM residual_mode: {residual_mode}")
        self.residual_mode = residual_mode
        hidden = max(channels // reduction, 8)

        self.mag_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )
        self.phase_res = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Tanh(),
        )
        self.mag_scale = nn.Parameter(torch.zeros(1))
        self.phase_scale = nn.Parameter(torch.zeros(1))
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self._eps = 1e-3

        for module in [self.mag_gate, self.phase_res]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F_repr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(F_repr)
        pha = torch.angle(F_repr + self._eps)

        mag_delta = self.mag_gate(torch.log1p(mag))
        mag = mag * (1.0 + self.mag_scale * mag_delta)

        pha_delta = self.phase_res(pha)
        pha = pha + self.phase_scale * pha_delta

        F_mod = mag * torch.exp(1j * pha)
        freq_out = torch.fft.irfft2(F_mod, s=x.shape[-2:], norm="ortho")

        if self.residual_mode == "current":
            residual = freq_out
        elif self.residual_mode == "delta":
            residual = freq_out - x
        else:  # norm_delta
            residual = freq_out - x
            denom = residual.detach().flatten(1).norm(dim=1).clamp_min(1e-6)
            view_shape = [residual.shape[0]] + [1] * (residual.ndim - 1)
            residual = residual / denom.view(*view_shape)

        return x + self.residual_scale * residual


class MPLSegAFMBlock(nn.Module):
    """MPLSeg-style AFM: hard-coded active gate, InstanceNorm, no learnable scales.

    Args:
        in_ch: input channels
        mid_ch: mid channels (default: in_ch)
        gate_strength: suppression strength multiplier (0.0=off, 1.0=full)
    """

    def __init__(self, in_ch: int, mid_ch: int | None = None, gate_strength: float = 1.0):
        super().__init__()
        mid = mid_ch or in_ch
        self.gate_strength = gate_strength

        self.mp = nn.Sequential(
            nn.Conv2d(in_ch, mid // 4, 1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Sigmoid(),
            nn.Conv2d(mid // 4, mid // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Sigmoid(),
            nn.Conv2d(mid // 4, mid, 1, bias=False),
            nn.InstanceNorm2d(mid),
            nn.Sigmoid(),
        )
        self.pa = nn.Sequential(
            nn.Conv2d(in_ch, mid // 4, 1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Tanh(),
            nn.Conv2d(mid // 4, mid // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid // 4),
            nn.Tanh(),
            nn.Conv2d(mid // 4, mid, 1, bias=False),
            nn.InstanceNorm2d(mid),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.ones(1))
        self._eps = 1e-3

        for seq in [self.mp, self.pa]:
            for m in seq:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + self._eps)

        mag = mag * (1.0 - self.gate_strength * self.mp(torch.sigmoid(torch.log(mag + self._eps))))
        pha = pha + self.pa(pha)

        fr = mag * torch.exp(1j * pha)
        freq_out = torch.fft.irfft2(fr, s=x.shape[-2:], norm="ortho")
        freq_out = F.relu(freq_out, inplace=False)

        return x + self.residual_scale * freq_out


class MagOnlyAFMBlock(MPLSegAFMBlock):
    """MPLSeg AFM with magnitude gate only — phase is pass-through (identity)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + self._eps)
        mag = mag * (1.0 - self.gate_strength * self.mp(torch.sigmoid(torch.log(mag + self._eps))))
        # phase: pass-through (no modulation)
        fr = mag * torch.exp(1j * pha)
        freq_out = torch.fft.irfft2(fr, s=x.shape[-2:], norm="ortho")
        freq_out = F.relu(freq_out, inplace=False)
        return x + self.residual_scale * freq_out


class PhaseOnlyAFMBlock(MPLSegAFMBlock):
    """MPLSeg AFM with phase residual only — magnitude is pass-through (identity)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fr = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fr)
        pha = torch.angle(fr + self._eps)
        # magnitude: pass-through (no gating)
        pha = pha + self.pa(pha)
        fr = mag * torch.exp(1j * pha)
        freq_out = torch.fft.irfft2(fr, s=x.shape[-2:], norm="ortho")
        freq_out = F.relu(freq_out, inplace=False)
        return x + self.residual_scale * freq_out

class PassThroughFFT(nn.Module):
    """FFT pass-through control: rFFT -> iRFFT with no gate, ReLU, residual_scale=1.0."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fr = torch.fft.rfft2(x, norm="ortho")
        freq_out = torch.fft.irfft2(fr, s=x.shape[-2:], norm="ortho")
        freq_out = F.relu(freq_out, inplace=False)
        return x + self.residual_scale * freq_out



class MultiScaleAFM(nn.Module):
    """Multi-scale AFM using MPLSeg-style blocks, one per FPN level.

    Args:
        channels: list of FPN channel depths per level
        gate_strength: shared suppression strength, or a list of per-level strengths
            (shorter lists are broadcast: [0.6] applies 0.6 to all levels)
    """

    def __init__(self, channels: list[int], gate_strength: float | list[float] = 0.6):
        super().__init__()
        strengths = [gate_strength] * len(channels) if isinstance(gate_strength, (int, float)) else gate_strength
        if len(strengths) != len(channels):
            raise ValueError(
                f"gate_strength length ({len(strengths)}) does not match "
                f"channels length ({len(channels)})"
            )
        self.blocks = nn.ModuleDict({
            str(i): MPLSegAFMBlock(in_ch=c, gate_strength=s)
            for i, (c, s) in enumerate(zip(channels, strengths))
        })

    def forward(self, feature_map: torch.Tensor, level: int) -> torch.Tensor:
        return self.blocks[str(level)](feature_map)


def build_afm_block(afm_type: str, channels: int, residual_mode: str = "current") -> nn.Module | None:
    if afm_type == "none":
        return None
    if afm_type == "old":
        return OldAFMBlock(channels=channels)
    if afm_type == "identity":
        return AFMBlock(channels=channels, residual_mode=residual_mode)
    if afm_type == "mplseg":
        return MPLSegAFMBlock(in_ch=channels)
    if afm_type == "mplseg_weak":
        return MPLSegAFMBlock(in_ch=channels, gate_strength=0.3)
    if afm_type == "mplseg_mid":
        return MPLSegAFMBlock(in_ch=channels, gate_strength=0.6)
    if afm_type == "mplseg_frozen":
        afm = MPLSegAFMBlock(in_ch=channels)
        for p in afm.mp.parameters():
            p.requires_grad = False
        for p in afm.pa.parameters():
            p.requires_grad = False
        return afm
    if afm_type == "mplseg_notune":
        return PassThroughFFT(in_ch=channels)
    if afm_type == "mplseg_mag_only":
        return MagOnlyAFMBlock(in_ch=channels, gate_strength=0.6)
    if afm_type == "mplseg_phase_only":
        return PhaseOnlyAFMBlock(in_ch=channels, gate_strength=0.6)
    raise ValueError(f"Unknown afm_type: {afm_type}")


MicroAFM = AFMBlock  # backward compatible alias
