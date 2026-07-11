"""Minimal additive endpoint for detector-only set-quality estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.ops import box_iou

from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import (
    DenseTeacherComponents,
)


@dataclass(frozen=True)
class RobustTeacherStats:
    median: torch.Tensor
    iqr: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "RobustTeacherStats":
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[0] < 2 or tensor.shape[1] != 3:
            raise ValueError("teacher values must have shape (N, 3), N >= 2")
        if not torch.isfinite(tensor).all():
            raise ValueError("teacher values must be finite")
        quantiles = torch.quantile(
            tensor,
            torch.tensor([0.25, 0.5, 0.75], dtype=tensor.dtype, device=tensor.device),
            dim=0,
        )
        iqr = quantiles[2] - quantiles[0]
        if (iqr <= 0).any():
            raise ValueError("all reduced teacher components must have positive IQR")
        return cls(median=quantiles[1], iqr=iqr)

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=self.median.dtype, device=self.median.device)
        if tensor.shape[-1:] != (3,):
            raise ValueError("reduced teacher values must end in dimension 3")
        return (tensor - self.median) / self.iqr

    def quality(self, values: torch.Tensor) -> torch.Tensor:
        standardized = self.standardize(values)
        return standardized[..., 0] - standardized[..., 1] - standardized[..., 2]


def reduced_teacher_values(components: DenseTeacherComponents) -> torch.Tensor:
    """Return coverage, total-error, and duplicate quality coordinates."""
    ground_truth = max(int(components.ground_truth_count), 1)
    predictions = max(int(components.prediction_count), 1)
    duplicate_edges = max(int(components.duplicate_edge_count), 1)
    return torch.stack(
        (
            components.coverage / ground_truth,
            (components.background_risk + components.class_risk) / predictions,
            components.duplicate_risk / duplicate_edges,
        )
    )


def build_sparse_pair_features(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    image_size: tuple[int, int] | torch.Tensor,
    *,
    min_iou: float,
) -> torch.Tensor:
    """Build symmetric same-class pair features on a detector-defined graph."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    count = boxes.shape[0]
    if scores.shape != (count,) or labels.shape != (count,):
        raise ValueError("scores and labels must have shape (N,)")
    if not 0.0 <= float(min_iou) <= 1.0:
        raise ValueError("min_iou must be in [0, 1]")
    size = torch.as_tensor(image_size, dtype=boxes.dtype, device=boxes.device).flatten()
    if size.numel() != 2 or (size <= 0).any():
        raise ValueError("image_size must contain positive height and width")
    if count < 2:
        return boxes.new_zeros((0, 5))
    height, width = size
    scale = torch.stack((width, height, width, height)).clamp_min(1.0)
    normalized = boxes / scale
    overlaps = box_iou(normalized, normalized)
    upper = torch.triu(torch.ones_like(overlaps, dtype=torch.bool), diagonal=1)
    edges = upper & labels[:, None].eq(labels[None, :]) & overlaps.ge(float(min_iou))
    edge_indices = torch.nonzero(edges, as_tuple=False)
    if edge_indices.numel() == 0:
        return boxes.new_zeros((0, 5))
    left, right = edge_indices[:, 0], edge_indices[:, 1]
    centers = 0.5 * (normalized[:, :2] + normalized[:, 2:])
    areas = (normalized[:, 2:] - normalized[:, :2]).clamp_min(1e-8).prod(dim=1)
    return torch.stack(
        (
            overlaps[left, right],
            scores[left] * scores[right],
            (scores[left] - scores[right]).abs(),
            (centers[left] - centers[right]).square().sum(dim=1).sqrt(),
            (areas[left].log() - areas[right].log()).abs(),
        ),
        dim=1,
    )


@dataclass(frozen=True)
class DenseEndpointOutput:
    quality: torch.Tensor
    unary_contribution: torch.Tensor
    pair_contribution: torch.Tensor
    node_contributions: torch.Tensor
    pair_contributions: torch.Tensor


class DenseSetEnergyEndpoint(nn.Module):
    """Permutation-invariant mean-additive unary/pair set energy."""

    def __init__(self, node_dim: int, pair_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if node_dim <= 0 or pair_dim <= 0 or hidden_dim <= 0:
            raise ValueError("endpoint dimensions must be positive")
        self.node_dim = int(node_dim)
        self.pair_dim = int(pair_dim)
        self.hidden_dim = int(hidden_dim)
        self.unary = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.pair = nn.Sequential(
            nn.Linear(self.pair_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.identity_bias = nn.Parameter(torch.zeros(()))

    def forward(self, node_features: torch.Tensor, pair_features: torch.Tensor) -> DenseEndpointOutput:
        if node_features.ndim != 2 or node_features.shape[1] != self.node_dim:
            raise ValueError("node_features must have shape (N, node_dim)")
        if pair_features.ndim != 2 or pair_features.shape[1] != self.pair_dim:
            raise ValueError("pair_features must have shape (E, pair_dim)")
        node_values = self.unary(node_features).squeeze(1)
        pair_values = self.pair(pair_features).squeeze(1)
        unary = node_values.mean() if node_values.numel() else self.identity_bias * 0.0
        pair = pair_values.mean() if pair_values.numel() else self.identity_bias * 0.0
        return DenseEndpointOutput(
            quality=self.identity_bias + unary + pair,
            unary_contribution=unary,
            pair_contribution=pair,
            node_contributions=node_values / max(int(node_values.numel()), 1),
            pair_contributions=pair_values / max(int(pair_values.numel()), 1),
        )
