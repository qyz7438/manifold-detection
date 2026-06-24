"""Small verifier / critic networks used by Plan 2.x runners.

These classes are copied from scripts/round274_v3_runner.py through
scripts/round281_runner.py.  Keeping them here lets runners import a single
implementation instead of duplicating the architecture in every script.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaseVerifier(nn.Module):
    """Spatial + geometric verifier used in reward-aligned RLVR (round274+)."""

    def __init__(self, roi_dim: int, geo_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, roi_feat: torch.Tensor, geo_feat: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.roi_net(roi_feat), self.geo_net(geo_feat)], dim=1)).squeeze(-1)


class FFTResidualVerifier(nn.Module):
    """FFT residual verifier predicting the component of reward unexplained by ROI."""

    def __init__(self, fft_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fft_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, fft_feat: torch.Tensor) -> torch.Tensor:
        return self.net(fft_feat).squeeze(-1)


class AlignedVerifier(nn.Module):
    """ROI + FFT + geometry verifier with Sigmoid output (round276+)."""

    def __init__(self, roi_dim: int, fft_dim: int, geo_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.roi_net = nn.Sequential(nn.Linear(roi_dim, hidden), nn.ReLU())
        self.fft_net = nn.Sequential(nn.Linear(fft_dim, hidden), nn.ReLU())
        self.geo_net = nn.Sequential(nn.Linear(geo_dim, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 32, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, roi_feat: torch.Tensor, fft_feat: torch.Tensor, geo_feat: torch.Tensor) -> torch.Tensor:
        r = self.roi_net(roi_feat)
        f = self.fft_net(fft_feat)
        g = self.geo_net(geo_feat)
        return self.head(torch.cat([r, f, g], dim=1)).squeeze(-1)


class PerChanFFTVerifier(nn.Module):
    """Per-channel FFT verifier with Sigmoid output (round267+)."""

    def __init__(self, fft_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fft_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ROIVerifier(nn.Module):
    """Simple ROI-feature verifier with Sigmoid output (round262/263)."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)


class FFTVerifier(nn.Module):
    """ROI + FFT concatenated verifier with Sigmoid output (round263)."""

    def __init__(self, roi_dim: int, fft_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(roi_dim + fft_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, roi: torch.Tensor, fft_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([roi.flatten(1), fft_feat], dim=1)).squeeze(-1)


def build_geo_features(boxes: torch.Tensor, image_shapes: list[tuple[int, int]],
                       img_map: list[int]) -> torch.Tensor:
    """Build normalized geometric features used by AlignedVerifier.

    Args:
        boxes: (N, 4) decoded boxes.
        image_shapes: list of (H, W) per image.
        img_map: list mapping each row of `boxes` to an image index.

    Returns:
        (N, 4) tensor with [cx_norm, cy_norm, log(w), log(h)].
    """
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    shapes_t = torch.tensor(image_shapes, device=boxes.device)
    cx_norm = cx / shapes_t[img_map, 1].clamp_min(1.0)
    cy_norm = cy / shapes_t[img_map, 0].clamp_min(1.0)
    return torch.stack([cx_norm, cy_norm, torch.log(w), torch.log(h)], dim=1)
