from __future__ import annotations

import torch


def radial_profile(amplitude: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
    if amplitude.ndim != 2:
        raise ValueError("amplitude must have shape [H, W].")
    height, width = amplitude.shape
    y, x = torch.meshgrid(torch.arange(height, device=amplitude.device), torch.arange(width, device=amplitude.device), indexing="ij")
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius = radius / radius.max().clamp(min=1e-6)
    bins = torch.clamp((radius * num_bins).long(), max=num_bins - 1)
    profile = torch.zeros((num_bins,), device=amplitude.device, dtype=amplitude.dtype)
    counts = torch.zeros((num_bins,), device=amplitude.device, dtype=amplitude.dtype)
    profile.scatter_add_(0, bins.flatten(), amplitude.flatten())
    counts.scatter_add_(0, bins.flatten(), torch.ones_like(amplitude).flatten())
    profile = profile / counts.clamp(min=1.0)
    return profile / profile.norm(p=2).clamp(min=1e-6)
