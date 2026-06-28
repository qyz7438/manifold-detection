"""Non-spectral FPN-level control adapter.

This module mirrors ``FPNSpectralManifold`` in shape and interface but uses a
real-valued bottleneck convolution instead of FFT + complex adapter.  It serves
as a control to disentangle the contribution of "low-rank feature adapter" from
"spectral/frequency-domain operations".

    FPN feature map (real, B, C, H, W)
        -> 1x1 conv  C -> latent
        -> ReLU
        -> 1x1 conv  latent -> C
        -> x' = x + alpha[key] * adapter(x)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNRealAdapter(nn.Module):
    """Real-valued low-rank FPN adapter for ablation.

    Args:
        channels: channel depth of every FPN level.
        num_levels: used only to derive default ``level_keys``.
        level_keys: explicit ordered FPN keys.
        latent_dim: bottleneck dimension.  Defaults to ``channels // 4``.
        init_alpha: initial per-level residual strength.
    """

    def __init__(
        self,
        channels: int = 256,
        num_levels: int = 4,
        level_keys: list[str] | None = None,
        latent_dim: int | None = None,
        init_alpha: float = 1e-3,
    ):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.level_keys = level_keys or [str(i) for i in range(num_levels)]
        self.latent_dim = latent_dim or max(1, channels // 4)

        # Shared bottleneck adapter across levels.
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, self.latent_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, channels, kernel_size=1, bias=False),
        )

        self.alpha = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
            for key in self.level_keys
        })

        self._stats_buffer: dict[str, list[torch.Tensor]] = {}
        self._last_stats: dict[str, float] = {}

    def reset_stats(self) -> None:
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

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        missing = set(self.level_keys) - set(features.keys())
        if missing:
            raise KeyError(
                f"FPNRealAdapter configured levels {self.level_keys} "
                f"missing from input keys {list(features.keys())}; missing={missing}"
            )

        out: dict[str, torch.Tensor] = {}
        step_stats: dict[str, torch.Tensor] = {}
        for key in self.level_keys:
            feat = features[key]
            b, c, h, w = feat.shape
            if c != self.channels:
                raise ValueError(
                    f"FPN level {key} has {c} channels, expected {self.channels}"
                )

            delta = self.adapter(feat)
            alpha = self.alpha[key]
            out[key] = feat + alpha * delta

            with torch.no_grad():
                feat_norm = feat.float().norm()
                delta_norm = delta.float().norm()
                step_stats[f"fpn_real/{key}_alpha"] = alpha.detach()
                step_stats[f"fpn_real/{key}_raw_rel_delta"] = delta_norm / (feat_norm + 1e-6)
                step_stats[f"fpn_real/{key}_eff_rel_delta"] = (alpha.abs() * delta_norm) / (feat_norm + 1e-6)
                step_stats[f"fpn_real/{key}_cos_delta_x"] = F.cosine_similarity(
                    delta.reshape(-1), feat.reshape(-1), dim=0
                )

        self._accumulate_stats(step_stats)
        return out
