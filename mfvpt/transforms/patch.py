from __future__ import annotations

import torch


def _sample_locations(batch: int, height: int, width: int, patch_size: int, device: torch.device):
    if patch_size >= min(height, width):
        raise ValueError("patch_size must be smaller than image height and width.")
    tops = torch.randint(0, height - patch_size + 1, (batch,), device=device)
    lefts = torch.randint(0, width - patch_size + 1, (batch,), device=device)
    return tops, lefts


def add_random_patch(x: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    batch, channels, height, width = x.shape
    out = x.clone()
    tops, lefts = _sample_locations(batch, height, width, patch_size, x.device)
    for i in range(batch):
        patch = torch.rand((channels, patch_size, patch_size), device=x.device, dtype=x.dtype)
        out[i, :, tops[i] : tops[i] + patch_size, lefts[i] : lefts[i] + patch_size] = patch
    return out


def add_checkerboard_patch(x: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    batch, channels, height, width = x.shape
    out = x.clone()
    tops, lefts = _sample_locations(batch, height, width, patch_size, x.device)
    yy, xx = torch.meshgrid(
        torch.arange(patch_size, device=x.device),
        torch.arange(patch_size, device=x.device),
        indexing="ij",
    )
    board = ((yy + xx) % 2).to(dtype=x.dtype).view(1, patch_size, patch_size).repeat(channels, 1, 1)
    for i in range(batch):
        out[i, :, tops[i] : tops[i] + patch_size, lefts[i] : lefts[i] + patch_size] = board
    return out


def add_qr_like_patch(x: torch.Tensor, patch_size: int = 32, block_size: int = 4) -> torch.Tensor:
    batch, channels, height, width = x.shape
    out = x.clone()
    tops, lefts = _sample_locations(batch, height, width, patch_size, x.device)
    grid_size = max(1, patch_size // block_size)
    for i in range(batch):
        coarse = torch.randint(0, 2, (1, grid_size, grid_size), device=x.device, dtype=x.dtype)
        patch = coarse.repeat_interleave(block_size, dim=1).repeat_interleave(block_size, dim=2)
        patch = patch[:, :patch_size, :patch_size].repeat(channels, 1, 1)
        out[i, :, tops[i] : tops[i] + patch_size, lefts[i] : lefts[i] + patch_size] = patch
    return out


def add_patch(x: torch.Tensor, patch_type: str = "random", patch_size: int = 32) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("x must have shape [B, C, H, W].")
    if patch_type == "random":
        return add_random_patch(x, patch_size)
    if patch_type == "checkerboard":
        return add_checkerboard_patch(x, patch_size)
    if patch_type in {"qr", "qr_like", "qr-like"}:
        return add_qr_like_patch(x, patch_size)
    raise ValueError(f"Unknown patch_type: {patch_type}")
