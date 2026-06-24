from __future__ import annotations

import torch
from torch import nn


class SpectralQualityHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def predict_quality(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(features))


def quality_input_dim(cache_meta: dict, feature_mode: str) -> int:
    roi_dim = int(cache_meta.get("roi_feature_dim", 0))
    amp_dim = int(cache_meta.get("amp_bins", 32))
    structure_dim = int(cache_meta.get("structure_dim", 8))
    if feature_mode == "roi":
        return roi_dim
    if feature_mode == "amp":
        return amp_dim
    if feature_mode == "amp_structure":
        return amp_dim + structure_dim
    if feature_mode == "roi_amp":
        return roi_dim + amp_dim
    if feature_mode == "roi_amp_structure":
        return roi_dim + amp_dim + structure_dim
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def build_quality_features(sample: dict, feature_mode: str) -> torch.Tensor:
    parts = []
    if feature_mode in {"roi", "roi_amp", "roi_amp_structure"}:
        parts.append(sample["roi_features"].float())
    if feature_mode in {"amp", "amp_structure", "roi_amp", "roi_amp_structure"}:
        parts.append(sample["amp_profiles"].float())
    if feature_mode in {"amp_structure", "roi_amp_structure"}:
        parts.append(sample["structure_features"].float())
    if not parts:
        raise ValueError(f"No feature parts selected for mode: {feature_mode}")
    return torch.cat(parts, dim=1)


def normalize_r_amp(raw_r_amp: torch.Tensor, stats: dict) -> torch.Tensor:
    if stats.get("mode", "minmax") == "zscore":
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), 1e-6)
        return ((raw_r_amp - mean) / std * 0.25 + 0.5).clamp(0.0, 1.0)
    min_value = float(stats.get("min", 0.0))
    max_value = float(stats.get("max", 1.0))
    return ((raw_r_amp - min_value) / max(max_value - min_value, 1e-6)).clamp(0.0, 1.0)


def make_quality_targets(sample: dict, stats: dict) -> torch.Tensor:
    s_amp = normalize_r_amp(sample["raw_r_amp"].float(), stats)
    is_tp = sample["is_tp"].float()
    return (is_tp * sample["ious"].float() * s_amp).clamp(0.0, 1.0)


def pairwise_ranking_loss(logits: torch.Tensor, is_tp: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    positives = logits[is_tp.bool()]
    negatives = logits[~is_tp.bool()]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.sum() * 0.0
    return torch.relu(margin - positives[:, None] + negatives[None, :]).mean()
