"""FFT/iFFT signal utilities for segmentation masks.

These functions adapt the detection-side ROI spectral utilities in
``spectral_detection_posttrain.signals.fft`` to dense masks.  The core idea is
unchanged: convert an image region into the frequency domain, then compare
amplitude/phase structure between prediction and target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_gray(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape (C, H, W)")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    return image.float().mean(dim=0) * mask.float()


def _gray_with_optional_hann(roi: torch.Tensor, use_hann: bool = True) -> torch.Tensor:
    if roi.ndim != 2:
        raise ValueError("roi must have shape (H, W)")
    if not use_hann:
        return roi
    h, w = roi.shape
    window = torch.outer(torch.hann_window(h, device=roi.device), torch.hann_window(w, device=roi.device))
    return roi * window


def compute_fft_amplitude(roi: torch.Tensor, use_hann: bool = True) -> torch.Tensor:
    gray = _gray_with_optional_hann(roi, use_hann=use_hann)
    fft = torch.fft.fft2(gray, dim=(-2, -1))
    amp = torch.fft.fftshift(torch.abs(fft))
    amp = torch.log1p(amp)
    return (amp - amp.min()) / (amp.max() - amp.min()).clamp(min=1e-6)


def radial_amplitude_profile(roi: torch.Tensor, bins: int = 16) -> torch.Tensor:
    if roi.ndim == 3:
        roi = roi.float().mean(dim=0)
    fft = torch.fft.fftshift(torch.fft.fft2(roi, norm="ortho"))
    amp = torch.log1p(torch.abs(fft))
    h, w = amp.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=amp.device), torch.arange(w, device=amp.device), indexing="ij")
    radius = torch.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    radius = radius / radius.max().clamp_min(1e-6)
    values = []
    for idx in range(bins):
        mask = (radius >= idx / bins) & (radius < (idx + 1) / bins)
        values.append(amp[mask].mean() if mask.any() else amp.new_tensor(0.0))
    return torch.stack(values)


def compute_amplitude_profile(image: torch.Tensor, mask: torch.Tensor, num_bins: int = 32, use_hann: bool = True) -> torch.Tensor:
    roi = _masked_gray(image, mask)
    return radial_amplitude_profile(_gray_with_optional_hann(roi, use_hann=use_hann), bins=num_bins)


def compute_lowfreq_phase_stats(image: torch.Tensor, mask: torch.Tensor, radius_ratio: float = 0.25, use_hann: bool = True) -> torch.Tensor:
    roi = _masked_gray(image, mask)
    gray = _gray_with_optional_hann(roi, use_hann=use_hann)
    phase = torch.fft.fftshift(torch.angle(torch.fft.fft2(gray, dim=(-2, -1))))
    height, width = phase.shape
    y, x = torch.meshgrid(torch.arange(height, device=phase.device), torch.arange(width, device=phase.device), indexing="ij")
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    selected = phase[radius <= radius.max().clamp(min=1e-6) * radius_ratio]
    if selected.numel() == 0:
        return torch.zeros((4,), dtype=image.dtype, device=image.device)
    cos_phase = selected.cos()
    sin_phase = selected.sin()
    return torch.stack(
        [
            cos_phase.mean(),
            cos_phase.std(unbiased=False),
            sin_phase.mean(),
            sin_phase.std(unbiased=False),
        ]
    )


def _normalized_sobel_magnitude(roi: torch.Tensor) -> torch.Tensor:
    gray = roi.mean(dim=0, keepdim=True).unsqueeze(0) if roi.ndim == 3 else roi.unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=roi.dtype,
        device=roi.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=roi.dtype,
        device=roi.device,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(gray, sobel_y, padding=1).squeeze()
    mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    return (mag - mag.min()) / (mag.max() - mag.min()).clamp(min=1e-6)


def phase_correlation_score(image: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor, use_hann: bool = True) -> torch.Tensor:
    pred_gray = _gray_with_optional_hann(_masked_gray(image, pred_mask), use_hann=use_hann)
    target_gray = _gray_with_optional_hann(_masked_gray(image, target_mask), use_hann=use_hann)
    pred_fft = torch.fft.fft2(pred_gray, dim=(-2, -1))
    target_fft = torch.fft.fft2(target_gray, dim=(-2, -1))
    cross_power = pred_fft * torch.conj(target_fft)
    cross_power = cross_power / cross_power.abs().clamp(min=1e-6)
    corr = torch.fft.ifft2(cross_power, dim=(-2, -1)).abs()
    return corr.max().clamp(0.0, 1.0)


def edge_similarity_score(image: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    pred_edge = _normalized_sobel_magnitude(_masked_gray(image, pred_mask)).flatten()
    target_edge = _normalized_sobel_magnitude(_masked_gray(image, target_mask)).flatten()
    cosine = F.cosine_similarity(pred_edge, target_edge, dim=0).clamp(-1.0, 1.0)
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def lowfreq_phase_similarity(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    radius_ratio: float = 0.25,
    use_hann: bool = True,
) -> torch.Tensor:
    pred_stats = compute_lowfreq_phase_stats(image, pred_mask, radius_ratio=radius_ratio, use_hann=use_hann)
    target_stats = compute_lowfreq_phase_stats(image, target_mask, radius_ratio=radius_ratio, use_hann=use_hann)
    mse = torch.mean((pred_stats - target_stats).square())
    return torch.exp(-mse).clamp(0.0, 1.0)


def compute_structure_similarity(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    phase_weight: float = 0.45,
    edge_weight: float = 0.35,
    lowfreq_weight: float = 0.20,
) -> torch.Tensor:
    phase = phase_correlation_score(image, pred_mask, target_mask)
    edge = edge_similarity_score(image, pred_mask, target_mask)
    lowfreq = lowfreq_phase_similarity(image, pred_mask, target_mask)
    total_weight = max(phase_weight + edge_weight + lowfreq_weight, 1e-6)
    score = (phase_weight * phase + edge_weight * edge + lowfreq_weight * lowfreq) / total_weight
    return score.clamp(0.0, 1.0)


def spectral_profile_similarity(image: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
    pred_profile = compute_amplitude_profile(image, pred_mask, num_bins=num_bins)
    target_profile = compute_amplitude_profile(image, target_mask, num_bins=num_bins)
    cosine = F.cosine_similarity(pred_profile, target_profile, dim=0).clamp(-1.0, 1.0)
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
