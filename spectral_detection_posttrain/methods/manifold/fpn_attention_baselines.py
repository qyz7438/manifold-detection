"""FPN-level channel attention baselines for controlled comparison.

Implements lightweight plug-in modules that can be inserted at the same location
as ``FPNSpectralManifold`` (between backbone and RPN/ROI heads) so that any
gain/loss can be attributed to the attention mechanism rather than to where it
is placed.

Supported mechanisms:
- SE: Squeeze-and-Excitation (global average pooling + MLP).
- FcaNet: frequency channel attention using a small fixed set of 2-D DCT
  coefficients instead of only the DC component.
- ECA: Efficient Channel Attention (1-D convolution over the channel dimension).
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dct_ii_matrix(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix of size ``(n, n)``."""
    k = torch.arange(n, dtype=torch.float32, device=device).view(n, 1)
    i = torch.arange(n, dtype=torch.float32, device=device).view(1, n)
    scale = math.sqrt(2.0 / n)
    mat = scale * torch.cos(math.pi / n * (i + 0.5) * k)
    mat[0, :] /= math.sqrt(2.0)
    return mat


def _zigzag_indices(n: int, k: int) -> list[tuple[int, int]]:
    """Return the first ``k`` (u, v) positions in a ``n x n`` DCT zigzag scan."""
    pairs = [(u, v) for u in range(n) for v in range(n)]
    pairs.sort(key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else -p[1]))
    return pairs[:k]


class FPNSE(nn.Module):
    """Squeeze-and-Excitation applied independently to every FPN level."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FPNFcaNet(nn.Module):
    """Simplified FcaNet-style frequency channel attention for FPN features.

    Each FPN level is first adaptively pooled to a fixed spatial size
    (``pool_size x pool_size``), then a 2-D DCT-II is applied.  The first
    ``freq_dim`` coefficients in zigzag order are fed to an MLP that outputs a
    per-channel weight.  This keeps the module resolution-agnostic and gives a
    fair comparison to the FFT-based ``FPNSpectralManifold``.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        freq_dim: int | None = None,
        pool_size: int = 7,
    ):
        super().__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.freq_dim = min(
            freq_dim or max(1, channels // reduction), pool_size * pool_size
        )
        mid = max(1, channels // reduction)

        # Precompute 2-D DCT basis and low-frequency selector.
        dct_mat = _dct_ii_matrix(pool_size)  # (P, P)
        # basis_2d[u, v, h, w] = dct_mat[u, h] * dct_mat[v, w]
        basis_2d = torch.einsum("uh,vw->uvhw", dct_mat, dct_mat)
        zigzag = _zigzag_indices(pool_size, self.freq_dim)
        basis_selected = torch.stack([basis_2d[u, v] for u, v in zigzag], dim=0)
        self.register_buffer("dct_basis", basis_selected)  # (F, P, P)

        # Shared 1-D conv over the frequency dimension (applied identically
        # to each channel) reduces the F frequency responses to a scalar weight.
        self.fc = nn.Sequential(
            nn.Conv1d(self.freq_dim, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # 1) Adaptive spatial pooling to fixed size.
        y = F.adaptive_avg_pool2d(x, self.pool_size)  # (B, C, P, P)
        # 2) 2-D DCT-II: project onto the first F low-frequency basis functions.
        z = torch.einsum("fhw,bchw->bcf", self.dct_basis, y)  # (B, C, F)
        # 3) Shared frequency-MLP -> per-channel weight.
        w = self.fc(z.transpose(1, 2))  # (B, F, C) -> (B, 1, C)
        w = w.transpose(1, 2).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * w


class FPNECA(nn.Module):
    """Efficient Channel Attention applied independently to every FPN level."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs(math.log(channels, 2) + b) / gamma)
        k = k if k % 2 == 1 else k + 1
        self.k = max(3, k)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.k,
            padding=(self.k - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.pool(x).squeeze(-1)  # (B, C, 1)
        y = y.transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * self.sigmoid(y)


class FPNAttentionWrapper(nn.Module):
    """Wrap per-level attention modules so they can be patched into a backbone.

    Args:
        attention_type: one of ``se``, ``fcanet``, ``eca``.
        channels: channel depth of each FPN level.
        level_keys: ordered FPN keys.
        reduction: MLP reduction ratio (SE/FcaNet).
    """

    def __init__(
        self,
        attention_type: Literal["se", "fcanet", "eca"],
        channels: int,
        level_keys: list[str],
        reduction: int = 16,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.level_keys = level_keys
        self.channels = channels
        self.reduction = reduction

        cls = {"se": FPNSE, "fcanet": FPNFcaNet, "eca": FPNECA}[attention_type]
        kwargs = {"channels": channels}
        if attention_type in ("se", "fcanet"):
            kwargs["reduction"] = reduction
        self.blocks = nn.ModuleDict(
            {key: cls(**kwargs) for key in level_keys}
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: self.blocks[key](features[key]) for key in self.level_keys}
