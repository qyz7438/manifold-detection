from __future__ import annotations

import torch


def _center_low_mask(height: int, width: int, ratio: float, device: torch.device) -> torch.Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1].")
    mask = torch.zeros((height, width), device=device)
    cy, cx = height // 2, width // 2
    rh = max(1, int(height * ratio / 2))
    rw = max(1, int(width * ratio / 2))
    mask[cy - rh : cy + rh, cx - rw : cx + rw] = 1.0
    return mask


def low_pass_filter(x: torch.Tensor, ratio: float = 0.25) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("x must have shape [B, C, H, W].")
    fft = torch.fft.fft2(x, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    _, _, height, width = x.shape
    mask = _center_low_mask(height, width, ratio, x.device).view(1, 1, height, width)
    filtered = fft_shift * mask
    out = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)).real
    return out.clamp(0.0, 1.0)


def high_freq_perturb(x: torch.Tensor, strength: float = 0.10, ratio: float = 0.25) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("x must have shape [B, C, H, W].")
    if strength < 0:
        raise ValueError("strength must be non-negative.")
    fft = torch.fft.fft2(x, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    _, _, height, width = x.shape
    low_mask = _center_low_mask(height, width, ratio, x.device)
    high_mask = (1.0 - low_mask).view(1, 1, height, width).bool()
    scale = 1.0 + strength * torch.randn_like(fft_shift.real)
    perturbed = torch.where(high_mask, fft_shift * scale, fft_shift)
    out = torch.fft.ifft2(torch.fft.ifftshift(perturbed, dim=(-2, -1)), dim=(-2, -1)).real
    return out.clamp(0.0, 1.0)
